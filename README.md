## SD-IDD: Selective Distillation for Incremental Defect Detec-tion



###  Introduction
The surface defects of industrial production are complex and diverse. Hence, trained de-fect detection models based on deep learning must consistently adapt to newly emerging defect categories. The trained models generally suffer from catastrophic forgetting as it learns new defect categories. To address this issue, we proposes a selective distillation for incremental defect detection (SD-IDD) model based on GFLv1. Specifically, three selective distillation strategies are proposed, including high-confidence classification distillation, dual-stage cascade regression distillation, and Intersection over Union (IoU)-driven diffi-culty-aware feature distillation. The high-confidence classification distillation aims to preserve critical discriminative knowledge of old categories within semantic confusion re-gions of the classification head, reducing interference from low-value regions. Dual-stage cascade regression distillation focuses on high-quality anchors through geometric prior coarse filtering and statistical fine filtering, utilizing IoU-weighted KL divergence distilla-tion loss to accurately transfer localization knowledge. IoU-driven difficulty-aware feature distillation adaptively allocates distillation resources, prioritizing features of high-difficulty targets. These selective distillation strategies significantly mitigate cata-strophic forgetting while enhancing the detection accuracy of new classes, without relying on old samples. Experimental results demonstrate that SD-IDD respectively achieves mAP_old values of 58.2% and 99.3%, mAP_new values of 69% and 97.3%, and mAP_all values of 63.6% and 98.3% on the NEU-DET datasets and DeepPCB datasets, outperform-ing existing incremental detection methods. 

<p align='left'>
  <img src='figs/SD-IDD.tif' width='721'/>
</p>


