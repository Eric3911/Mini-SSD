python object_detection/export_inference_graph.py 
         --input_type image_tensor
         --pipeline_config_path configs/faster_rcnn_resnet101.config 
         --trained_checkpoint_prefix model.ckpt-20000
         --output_directory output_inference_graph
