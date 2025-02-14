import numpy as np
import os
import json
import shutil
import torch

from mmpose.apis import inference_bottom_up_pose_model

from xtcocotools.coco import COCO
from utils.metrics.coco_similarity import COCOevalSimilarity
from mmpose.apis import init_pose_model
#----------------------------------------------------------------------------

class PoseMetrics():
    def __init__(self,
                 device,
                 mmpose_config_file,
                 mmpose_checkpoint_file,
                 run_name) -> None:
        self.mmpose_config_file=mmpose_config_file
        self.mmpose_checkpoint_file=mmpose_checkpoint_file
        self.device=device
        self.mmpose_model=init_pose_model(mmpose_config_file, mmpose_checkpoint_file, device=self.device)
        self.mmpose_model.eval()
        self.run_name = run_name
        
    def __call__(self,batch, output_images):
        pose_result=self.compute(batch, output_images)
        return pose_result
        

    def compute(self,
                batch, 
                output_images):    

        gt_pose=batch["pose"]
        b,h,w,c=output_images.shape
        gt_pose_results={
                            "images":[],
                            "annotations":[],
                            "categories": [{'id': 1, 'name': 'person'}]
                        }
        dt_pose_results=[]
        
        for idx in range(b):
            gt_pose_results['images'].append({"file_name":"None",
                                "height":h,
                                "width":w,
                                "id":idx,
                                "page_url":"None",
                                "image_url":"None",
                                "picture_name":"None",
                                "author":"None",
                                "description":"None",
                                "category":"None"
                                })
            if gt_pose.is_cuda:
                gt_pose = gt_pose.cpu().detach()
            present_annotation_info=np.array(gt_pose[idx,...])
            for anno_i in range(present_annotation_info.shape[0]):
                present_annotation=present_annotation_info[anno_i,:,:]
                keypoint_num=len(np.where(present_annotation[:,0]>0)[0])
                if keypoint_num:
                    gt_pose_results['annotations'].append({
                        "keypoints":list(present_annotation.reshape(-1)),
                        "num_keypoints":keypoint_num,
                        "iscrowd": 0,
                        "image_id": idx,
                        "category_id": 1,
                        "id": idx*10+anno_i,
                        "bbox": [
                            min(present_annotation[:,0]),
                            min(present_annotation[:,1]),
                            max(present_annotation[:,0])-min(present_annotation[:,0]),
                            max(present_annotation[:,1])-min(present_annotation[:,1])
                        ],
                        "area":(max(present_annotation[:,1])-min(present_annotation[:,1]))*(max(present_annotation[:,0])-min(present_annotation[:,0]))
                    })
                    
            present_image=output_images[idx,...].clone().detach()
            if len(torch.where(present_image.reshape(-1)>1)[0])==0:
                present_image*=255
                present_image = present_image.cpu().detach().numpy()
                present_image = present_image.astype(np.uint8)
            
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                pose_results, returned_outputs = inference_bottom_up_pose_model(self.mmpose_model, present_image,pose_nms_thr=1., )
                # pose_results, returned_outputs = inference_bottom_up_pose_model(self.mmpose_model, present_image,pose_nms_thr=1., outputs=['backbone'])
                # print(returned_outputs)
                # print(len(returned_outputs))
                # print(returned_outputs[0]['backbone'][0].shape)
                # print(pose_results)
                # x = returned_outputs[0]['keypoint_head']
                # print(x)
                # print(self.mmpose_model.keypoint_head.loss.ae_loss(x,x))

            if len(pose_results):
                for pose_result in pose_results:
                    dt_pose_results.append({
                        "category_id": 1,
                        "image_id": idx,
                        "keypoints":[content.item() for content in list(pose_result["keypoints"].reshape(-1))],
                        "score":pose_result["score"].item()
                    })
        # ensure that the tmp dir name is unique
        import time, random
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        rand_num = random.randint(0, 999999)
        tmp_res_dir = os.path.join('tmp', self.run_name + '_' + current_time + str(rand_num)) 
        if not os.path.exists(tmp_res_dir):
            os.mkdir(tmp_res_dir)
        gt_file_path=os.path.join(tmp_res_dir,"gt_keypoints.json")
        with open(gt_file_path,"w") as f:
            json.dump(gt_pose_results,f)
               
        dt_file_path=os.path.join(tmp_res_dir,"dt_keypoints.jsons")
        with open(dt_file_path,"w") as f:
            json.dump(dt_pose_results,f)
        
        gt_coco = COCO(gt_file_path)
        dt_coco = gt_coco.loadRes(dt_file_path)
        coco_eval = COCOevalSimilarity(gt_coco, dt_coco, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_ar_result={
                "Distance Average Precision    (DAP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]":coco_eval.stats[0],
                "Distance Average Precision    (DAP) @[ IoU=0.50      | area=   all | maxDets= 20 ]":coco_eval.stats[1],
                "Distance Average Precision    (DAP) @[ IoU=0.75      | area=   all | maxDets= 20 ]":coco_eval.stats[2],
                "Distance Average Precision    (DAP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]":coco_eval.stats[3],
                "Distance Average Precision    (DAP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]":coco_eval.stats[4],
                "Distance Average Recall       (DAR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]":coco_eval.stats[5],
                "Distance Average Recall       (DAR) @[ IoU=0.50      | area=   all | maxDets= 20 ]":coco_eval.stats[6],
                "Distance Average Recall       (DAR) @[ IoU=0.75      | area=   all | maxDets= 20 ]":coco_eval.stats[7],
                "Distance Average Recall       (DAR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]":coco_eval.stats[8],
                "Distance Average Recall       (DAR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]":coco_eval.stats[9],
            }
        
        coco_eval.evaluateSimilarity()
        coco_eval.accumulateSimilarity()
        coco_eval.summarizeSimilarity()
        
        cosine_silimarity_result={
                "Similarity Average Precision  (SAP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]":coco_eval.statsSimilarity[0],
                "Similarity Average Precision  (SAP) @[ IoU=0.50      | area=   all | maxDets= 20 ]":coco_eval.statsSimilarity[1],
                "Similarity Average Precision  (SAP) @[ IoU=0.75      | area=   all | maxDets= 20 ]":coco_eval.statsSimilarity[2],
                "Similarity Average Precision  (SAP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]":coco_eval.statsSimilarity[3],
                "Similarity Average Precision  (SAP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]":coco_eval.statsSimilarity[4],
                "Similarity Average Recall     (SAR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]":coco_eval.statsSimilarity[5],
                "Similarity Average Recall     (SAR) @[ IoU=0.50      | area=   all | maxDets= 20 ]":coco_eval.statsSimilarity[6],
                "Similarity Average Recall     (SAR) @[ IoU=0.75      | area=   all | maxDets= 20 ]":coco_eval.statsSimilarity[7],
                "Similarity Average Recall     (SAR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]":coco_eval.statsSimilarity[8],
                "Similarity Average Recall     (SAR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]":coco_eval.statsSimilarity[9],
            }
        
        human_number_diff=[]
        for img_idx in range(len(gt_coco.imgToAnns)):
            
            human_number_diff.append(abs(len(gt_coco.imgToAnns[img_idx])-len(dt_coco.imgToAnns[img_idx])))
        
        human_number_diff_result={
                "Human Number Difference       (HND)                                               ": np.mean(human_number_diff).item()
        }
        
        results={**ap_ar_result,**cosine_silimarity_result,**human_number_diff_result}
             
        try:
            if os.path.exists(tmp_res_dir):
                shutil.rmtree(tmp_res_dir)
        except:
            print('Exception!')

        return results
        

#----------------------------------------------------------------------------
