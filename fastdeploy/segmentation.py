import os
import logging
import boto3
import io
import json
import cv2
from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter import brush
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import fastdeploy as fd
import random
import string
import numpy as np


class MissingSegLables(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


logger = logging.getLogger(__name__)


class FastDeploySegmentation(LabelStudioMLBase):

    def __init__(self, checkpoint_dir, seg_labels, labels_file=None, image_dir=None, device='cpu', use_trt=False, **kwargs):
        super(FastDeploySegmentation, self).__init__(**kwargs)
        self.labels_file = labels_file
        self.checkpoint_dir = checkpoint_dir
        self.endpoint_url = kwargs.get('endpoint_url')
        if self.endpoint_url:
            logger.info(f'Using s3 endpoint url {self.endpoint_url}')

        # read seg labels from file
        if seg_labels and os.path.exists(seg_labels):
            self.seg_labels = json_load(seg_labels, True)
            logger.info('Using labels...')
            logger.info(self.seg_labels)
        else:
            raise MissingSegLables

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
            self.parsed_label_config, 'BrushLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        logger.info(f"Load new model from {self.checkpoint_dir}")

        # build option
        runtime_option = fd.RuntimeOption()
        if device == 'gpu':
            runtime_option.use_gpu()
            runtime_option.use_paddle_infer_backend()
            if use_trt:
                runtime_option.use_trt_backend()
                runtime_option.set_trt_cache_file(os.path.join(self.checkpoint_dir, "trt_cache.trt"))
                # If use original Tensorrt, not Paddle-TensorRT,
                # comment the following two lines
                runtime_option.enable_paddle_to_trt()
                runtime_option.enable_paddle_trt_collect_shape()

        # load model
        self.model = fd.vision.segmentation.PaddleSegModel(
            model_file=os.path.join(self.checkpoint_dir, "model.pdmodel"),
            params_file=os.path.join(self.checkpoint_dir, "model.pdiparams"),
            config_file=os.path.join(self.checkpoint_dir, "deploy.yaml"),
            runtime_option=runtime_option
        )

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

        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_width, img_height = get_image_size(image_path)

            pred = self.model.predict(img)
            mask = np.array(pred.label_map).reshape(img_height, img_width).astype(np.uint8)

            for label_id, label in self.seg_labels.items():
                label_mask = np.where(mask == label_id, label_id, 0)

                if len(np.unique(label_mask)) == 1:
                    continue

                output_label = self.label_map.get(label, label)
                if output_label not in self.labels_in_config:
                    logger.warning(f'{output_label} label not found in project config.')
                    continue

                label_mask = label_mask * 255
                rle = brush.mask2rle(label_mask)
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    # "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [output_label],
                    },
                    "type": "brushlabels",
                    "id": ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)), # creates a random ID for your label every time
                    "readonly": False,
                })

        return [{'result': results}]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
