"""Main API for computing and reporting pose&text2image metrics."""
from .pose_metrics import PoseMetrics
from .quality_metrics import QualityMetrics


#----------------------------------------------------------------------------

class PoseTxt2ImgMetrics():
    def __init__(self,
                 device,
                 pose,
                 quality,
                 ) -> None:
        
        # pose metrics
        self.device=device
        self.pose=pose
        self.quality=quality
        self.pose_metrics_calculator=PoseMetrics(self.device,**self.pose)
        
        
    def calc_metrics(self,batch, output,metrics=[""]):
        """
            metric_list: fid, kid, LPIPS, R-precision, CLIPSIM, pose_ap_ar, bbox_ap_ar, pose_cosine_similarity, human_num, structure_correctness
        """
        results_dict={}
        for metric in metrics:
            result_dict = self.calc_metric(metric=metric, batch=batch, output=output)
            results_dict[metric]=result_dict
            
        return results_dict
    
    def calc_metric(self,metric, batch, output): 
        if metric=="quality":
            # will calculate separately
            return {"None":0}
        
        # Calculate.
        results = eval("self."+metric+"_metrics_calculator")(batch, output)

        # Broadcast results.
        for key, value in list(results.items()):
            results[key] = value

        # Decorate with metadata.
        return results
    