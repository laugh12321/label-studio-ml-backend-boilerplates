import os
import logging
import boto3
import io
import json
from PIL import Image
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

class MissingYoloLables(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


logger = logging.getLogger(__name__)


class YOLODetectorModel(LabelStudioMLBase):

    def __init__(self,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 yolo_labels='labels.txt',
                 score_threshold=0.45,
                 iou_threshold=0.25,
                 device='cpu', **kwargs):
        """
        Load YoloV5 model from checkpoint into memory.
        Set mappings from yolov5 classes to target labels
        :param checkpoint_file: Absolute path to yolov5 serialized model
            repo_dir = "/home/projects/yolov5/runs/train/sku110k-640-fixed/weights/best.pt"
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from yolo labels to custom labels {"airplane": "Boeing"}
        :param yolo_labels: file with yolo label names, plain text with each label on a new line
        :param score_threshold: score threshold to remove predictions below one
        :param iou_threshold: IoU threshold for yolo NMS
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs: endpoint_url - endpoint URL for custom s3 storage
        """

        super(YOLODetectorModel, self).__init__(**kwargs)
        checkpoint_file = checkpoint_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        self.iou_threshold = iou_threshold
        self.score_thresh = score_threshold
        self.device = device
        self.endpoint_url = kwargs.get('endpoint_url')
        if self.endpoint_url:
            logger.info(f'Using s3 endpoint url {self.endpoint_url}')

        # read yolo labels from file
        if yolo_labels and os.path.exists(yolo_labels):
            with open(yolo_labels, 'r') as f:
                yolo_labels_list = f.readlines()
            yolo_labels_list = list(map(lambda x: x.strip(), yolo_labels_list))
            yolo_labels_list = list(map(lambda x: x if x[-1] != '\n' else x[:-1], yolo_labels_list))
            self.yolo_labels = yolo_labels_list
            logger.info('Using labels...')
            logger.info(", ".join(self.yolo_labels))
        else:
            raise MissingYoloLables

        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        # create a label map
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        logger.info(f"Load new model from {self.checkpoint_file}")
        self.model = torch.hub.load("ultralytics/yolov5", "custom",
                                    path=self.checkpoint_file, trust_repo=True)
        self.model.conf = self.score_thresh
        self.model.iou = self.iou_threshold
        print(f"confidence threshold {self.model.conf}")
        print(f"IoU threshold {self.model.iou}")

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            if self.endpoint_url:
                client = boto3.client('s3', endpoint_url=self.endpoint_url)
            else:
                client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as ex:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {ex}')
        return image_url

    def predict(self, tasks, **kwargs):

        results = []
        all_scores = []
        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)
            with torch.no_grad():
                preds = self.model(img, size=640)
            preds_df = preds.pandas().xyxy[0]
            for x_min, y_min, x_max, y_max, confidence, class_, label in zip(preds_df['xmin'], preds_df['ymin'],
                                                                             preds_df['xmax'], preds_df['ymax'],
                                                                             preds_df['confidence'], preds_df['class'],
                                                                             preds_df['name']):
                # add label name from label_map
                output_label = self.label_map.get(label, label)
                if output_label not in self.labels_in_config:
                    logger.warning(f'{output_label} label not found in project config.')
                    continue
                if confidence < self.score_thresh:
                    continue

                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x_min / img_width * 100,
                        'y': y_min / img_height * 100,
                        'width': (x_max - x_min) / img_width * 100,
                        'height': (y_max - y_min) / img_height * 100
                    },
                    'score': confidence
                })
                all_scores.append(confidence)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{
            'result': results,
            'score': avg_score
        }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
