# ############# DECODE OUTPUT ##############
import numpy as np
import tensorflow as tf


class RescaleOutput:
    def __init__(self, anchors):
        self.anchors = anchors

    def fit(self, output):
        img_grid_height, img_grid_width, img_box, _ = output.shape
        anchors_width, anchors_height = self.anchors[::2], self.anchors[1::2]

        arr_img_grid_height = np.zeros_like(output[..., 0])
        arr_img_grid_width = np.zeros_like(output[..., 0])
        arr_img_anchors_width = np.zeros_like(output[..., 0])
        arr_img_anchors_height = np.zeros_like(output[..., 0])

        for i in range(img_grid_height):
            arr_img_grid_height[i, :, :] = i

        for i in range(img_grid_width):
            arr_img_grid_width[:, i, :] = i
        
        for i in range(img_box):
            arr_img_anchors_width[:, :, i] = anchors_width[i]

        for i in range(img_box):
            arr_img_anchors_height[:, :, i] = anchors_height[i]

        # rescale x, y, width, height in range 0-1
        output[..., 0] = (tf.sigmoid(output[..., 0]).numpy() + arr_img_grid_width) / img_grid_width
        output[..., 1] = (tf.sigmoid(output[..., 1]).numpy() + arr_img_grid_height) / img_grid_height
        output[..., 2] = (np.exp(output[..., 2]) * arr_img_anchors_width) / img_grid_width
        output[..., 3] = (np.exp(output[..., 3]) * arr_img_anchors_height) / img_grid_height

        # rescale confidence in range 0-1
        output[..., 4]   = tf.sigmoid(output[..., 4]).numpy()

        # rescale class probability in range 0-1
        confidence_expanded      = np.expand_dims(output[..., 4], -1)
        output[..., 5:]  = confidence_expanded * tf.nn.softmax(output[..., 5:], axis=-1).numpy()

        return output

class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max, confidence=None, labels_probability=None):
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max
        self.confidence = confidence
        self.set_label(labels_probability)
        
    def set_label(self, labels_probability):
        self.labels_probability = labels_probability
        self.label = np.argmax(self.labels_probability)
    
    def get_label(self):
        return self.label
    
    def get_highest_label_probability_score(self):
        return self.labels_probability[self.label]

def get_image_boxes(rescaled_result, obj_threshold=0.2):
    img_grid_height, img_grid_width, img_fitted_anchor, _ = rescaled_result.shape
    print(" * rescaled_result.shape {}".format(rescaled_result.shape))
    img_boxes = [] # List of boxes that having confidence > obj_threshold
    for row in range(img_grid_height):
        for column in range(img_grid_width):
            for i in range(img_fitted_anchor):
                labels_probability = rescaled_result[row, column, i, 5:]
                
                if np.sum(labels_probability) > 0:
                    center_x, center_y, box_width, box_height = rescaled_result[row, column, i, :4]
                    confidence = rescaled_result[row, column, i, 4]
                    box = BoundingBox(x_min=center_x - (box_width / 2),
                            y_min=center_y - (box_height / 2),
                            x_max=center_x + (box_width / 2),
                            y_max=center_y + (box_height / 2),
                            confidence=confidence,
                            labels_probability=labels_probability
                        )
                    
                    # print("\n * {} > {} : {}".format(box.get_highest_label_probability_score().item(), obj_threshold, box.get_highest_label_probability_score().tolist() > obj_threshold))
                    # if box.get_highest_label_probability_score() > obj_threshold: print(" * Appending box")                    
                    if box.get_highest_label_probability_score() > obj_threshold: img_boxes.append(box)

    # print(" * img_boxes {}".format(img_boxes))
    return img_boxes

class AnchorBoxMatching:
  def __init__(self, anchors=None):
    if not (anchors is None):
      self.anchors = [BoundingBox(0, 0, anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
  
  def _calculate_intersection(self, box1, box2):
    x1_box1, x2_box1 = box1
    x1_box2, x2_box2 = box2

    if x1_box2 < x1_box1:
      if x2_box2 < x1_box1: return 0
      else: return min(x2_box1, x2_box2) - x1_box1
    else:
      if x2_box1 < x1_box2: return 0
      else: return min(x2_box1, x2_box2) - x1_box2
  
  def _calculate_box_area(self, box):
    box_width = box.x_max - box.x_min
    box_height = box.y_max - box.y_min
    return box_width * box_height
  
  def calculate_iou(self, box1, box2):
    intersection_width = self._calculate_intersection([box1.x_min, box1.x_max], [box2.x_min, box2.x_max])
    intersection_height = self._calculate_intersection([box1.y_min, box1.y_max], [box2.y_min, box2.y_max])
    intersection_area = intersection_width * intersection_height

    box1_area = self._calculate_box_area(box1)
    box2_area = self._calculate_box_area(box2)
    union_area = box1_area + box2_area - intersection_area

    return float(intersection_area) / union_area
  
  def fit(self, box_width, box_height):
    matched_anchor, max_iou = -1, -1

    for anchor_index in range(len(self.anchors)):
      iou = self.calculate_iou(BoundingBox(0, 0, box_width, box_height), self.anchors[anchor_index])
      if max_iou < iou: matched_anchor, max_iou = anchor_index, iou

    return matched_anchor, max_iou

def calculate_nonmax_suppression(img_boxes, iou_threshold=0.2, obj_threshold=0.2):
    total_boxes = len(img_boxes)
    total_label = len(img_boxes[0].labels_probability)
    anchorBoxMatching = AnchorBoxMatching()
    index_boxes = []

    for label_index in range(total_label):
        all_nth_label_probabilities = [img_box.labels_probability[label_index] for img_box in img_boxes]
        box_indices = list(np.argsort(all_nth_label_probabilities)[::-1])

        for i in range(total_boxes):
            ith_index_of_box_indices = box_indices[i]

            if img_boxes[ith_index_of_box_indices].labels_probability[label_index] == 0.: continue
            else:
                index_boxes.append(ith_index_of_box_indices)
                for j in range(i + 1, total_boxes):
                    jth_index_of_box_indices = box_indices[j]
                    
                    iou_i_j = anchorBoxMatching.calculate_iou(img_boxes[ith_index_of_box_indices], img_boxes[jth_index_of_box_indices])
                    if iou_i_j > iou_threshold:
                        img_boxes[jth_index_of_box_indices].labels_probability[label_index] = 0
                        img_boxes[jth_index_of_box_indices].set_label(img_boxes[jth_index_of_box_indices].labels_probability)
    
    final_img_boxes = []
    all_confidence = []
    for i in index_boxes:
        if img_boxes[i].get_highest_label_probability_score() > obj_threshold:
            if img_boxes[i].confidence not in all_confidence:
                all_confidence.append(img_boxes[i].confidence)
                final_img_boxes.append(img_boxes[i])

    return final_img_boxes

# '''
# def draw_boxes(image, img_boxes, labels):
#     image_h, image_w, _ = image.shape
    
#     adjust_boxes = lambda n, nmax: max(min(nmax, n), 0)
#     color_palette = list([tuple(np.random.choice(range(255), size=3) / 255.) for i in range(8)])
#     for box, color in zip(img_boxes, color_palette):
#         x_min = adjust_boxes(int(box.x_min * image_w), image_w)
#         y_min = adjust_boxes(int(box.y_min * image_h), image_h)
#         x_max = adjust_boxes(int(box.x_max * image_w), image_w)
#         y_max = adjust_boxes(int(box.y_max * image_h), image_h)

#         print(f'{labels[box.label]} {box.get_highest_label_probability_score() * 100}% [x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}]')
#         cv2.rectangle(image,
#                       pt1=(x_min,y_min), 
#                       pt2=(x_max,y_max), 
#                       color=color
#                       )
#         cv2.putText(img=image, 
#                     text=f'{labels[box.label]} {int(box.get_highest_label_probability_score() * 100)}%', 
#                     org=(x_min+ 13, y_min + 13),
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=1e-3 * image_h,
#                     color=(1, 0, 1)
#                     )     
#     return image
# '''



# ######## OUTPUT CONFIGURATION ###########
# LABELS = ['bicycle', 'bus', 'car', 'motorbike', 'person']
# GRID_H, GRID_W = 13, 13
# ANCHORS = np.array([0.07095013, 0.13790466, 0.74620075, 0.8126473, 0.37125614, 0.65841728, 0.18252735, 0.41417845])
# ANCHORS[::2], ANCHORS[1::2] = ANCHORS[::2] * GRID_W, ANCHORS[1::2] * GRID_H
# IMG_HEIGHT, IMG_WIDTH = 416, 416
# GROUNDTRUTH_BOX = 20
# obj_threshold = 0.2
# iou_threshold = 0.5






# ######## PREDICT OUTPUT ##############
# input_image_path = './gambar.jpg'
# #output_filename = 'gambar.jpg'
# img_boxes = []

# image = cv2.imread(input_image_path)
# image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
# image = image / 255.

# X = np.expand_dims(image, axis=0)
# Y = np.zeros((1, 1, 1, 1, GROUNDTRUTH_BOX, 4))
# output = model.predict([X, Y])

# rescaleResult = RescaleOutput(ANCHORS)
# rescaled_result = rescaleResult.fit(output[0])
# img_boxes = get_image_boxes(rescaled_result, obj_threshold)
# if img_boxes:
#     img_boxes = calculate_nonmax_suppression(img_boxes, iou_threshold, obj_threshold)
#     #image = draw_boxes(X[0], img_boxes, LABELS)
#     #cv2.imwrite(output_filename, image * 255.)

# adjust_boxes = lambda n, nmax: max(min(nmax, n), 0)

# final_output = []
# for i in range(len(img_boxes)):
#     final_output.append({
#         'label': LABELS[img_boxes[i].label],
#         'confidence': img_boxes[i].confidence,
#         'xmin': adjust_boxes(int(img_boxes[i].x_min * IMG_WIDTH), IMG_WIDTH),
#         'ymin': adjust_boxes(int(img_boxes[i].y_min * IMG_HEIGHT), IMG_HEIGHT),
#         'xmax': adjust_boxes(int(img_boxes[i].x_max * IMG_WIDTH), IMG_WIDTH),
#         'ymax': adjust_boxes(int(img_boxes[i].y_max * IMG_HEIGHT), IMG_HEIGHT)
#     })
