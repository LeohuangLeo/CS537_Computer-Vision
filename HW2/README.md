HW2 is about image retrieval by matching keypoints.

Given a query image, LaTeX: qq, and a large dataset of images, LaTeX: D=\left\{r_i\right\}D = { r i }, find in LaTeX: DD a subset of LaTeX: KK most similar images to LaTeX: qq , denoted as LaTeX: S_q(K)\subset DS q ( K ) ⊂ D. Similarity between LaTeX: qq and  LaTeX: r\in Dr ∈ D, denoted as LaTeX: s\left(q,r\right)s ( q , r ), is computed by matching their keypoints. Let LaTeX: f^q_kf k q denote a deep feature of LaTeX: kk-th keypoint in LaTeX: qq, and LaTeX: f^r_lf l r denote a deep feature of LaTeX: ll-th keypoint  in LaTeX: rr. Then, we define LaTeX: s\left(q,r\right)s ( q , r ) as a maximum total similarity of their matched pairs of keypoints:

 LaTeX: s\left(q,r\right)=\sum_{\left(f^q_k\:,\:f^r_l\right)\in M}\:s\left(f^q_k\:,\:\:f^r_l\right)s ( q , r ) = ∑ ( f k q , f l r ) ∈ M s ( f k q , f l r ), where LaTeX: s\left(f^q_k\:,\:\:f^r_l\right)  \;= \;s_{kl}\;=\;\frac{1}{2}\left(1+\frac{(f_k^q)^\top f_l^r}{\|f_k^q\|_2\|f_l^r\|_2}\right)\;\in\;[0,1]s ( f k q , f l r ) = s k l = 1 2 ( 1 + ( f k q ) ⊤ f l r ‖ f k q ‖ 2 ‖ f l r ‖ 2 ) ∈ [ 0 , 1 ]

In LaTeX: S_q(K)S q ( K ), there could be true and false positives. We say that an image LaTeX: r\in S_q(K)r ∈ S q ( K ) is a true positive if LaTeX: rr shows the same object as the query image LaTeX: qq. Otherwise, LaTeX: rr is a false positive. We will evaluate keypoint matching by estimating precision and recall of image retrieval, defined as

PrecisionLaTeX: \left(q,\:K\right)( q , K ) = LaTeX: P\left(q,\:K\right)P ( q , K ) = [number of true positives in LaTeX: S_q(K)S q ( K )]  /  K

RecallLaTeX: \left(q,\:K\right)( q , K )  = LaTeX: R\left(q,\:K\right)R ( q , K ) = [number of true positives in LaTeX: S_q(K)S q ( K )]  / (number of true positives in LaTeX: DD) 

 

Tasks:
Download the set LaTeX: DD of 136 images, and the set LaTeX: QQ of 34 query images from image_retrieval.zip.
For every image in LaTeX: DD and LaTeX: QQ, use your code from HW1 to:
Detect 20 strongest SIFT keypoints in the image;
Extract 20 image patches of size 32x32 pixels, centered at the detected SIFT keypoints, from the image;
For every image in LaTeX: DD and LaTeX: QQ, use our CNNPreview the document with these parameters (learned on the image patches of HW1) to compute 20 deep features LaTeX: ff of 20 32x32 image patches extracted in the previous step. If you prefer, instead of ours, you could also use your own best performing CNN from HW1 for computing the 20 deep features.  
For every query image LaTeX: q\in Qq ∈ Q
Match LaTeX: qq to every image LaTeX: r\in Dr ∈ D by matching the deep features of their keypoints, using:
Many-to-many matching:
Compute the similarity vector, LaTeX: {\bf  s}=\:\left[s_{kl}\right]s = [ s k l ], of matching feature pairs (LaTeX: f^q_kf k q, LaTeX: f^r_lf l r) from the two images;
Find the binary indicator of matches, LaTeX: {\bf x}^{*}x ∗, from the following optimization problem specified for a continuous vector LaTeX: {\bf x}x: maximize LaTeX: {\bf s}^\top {\bf x}s ⊤ x,   s.t. LaTeX: \|{\bf x}\|_2=1‖ x ‖ 2 = 1, and LaTeX: {\bf x}\ge {\bf 0}x ≥ 0
Compute the total similarity of matching: LaTeX: s^{\text{many2many}}\left(q,r\right)={\bf s}^\top{\bf x}^*s many2many ( q , r ) = s ⊤ x ∗
One-to-one matching:
Compute the cost matrix, LaTeX: C=\left[c_{kl}\right]C = [ c k l ], of matching feature pairs (LaTeX: f^q_kf k q, LaTeX: f^r_lf l r) from the two images, where  LaTeX: c_{kl}=1-s_{kl}c k l = 1 − s k l
Find LaTeX: X^*=[x_{kl}^*]X ∗ = [ x k l ∗ ] that minimizes the following constrained objective: minimize  trace(LaTeX: C^\top XC ⊤ X),  s.t. LaTeX: \forall l: \sum_k x_{kl} = 1∀ l : ∑ k x k l = 1 and LaTeX: \forall k: \sum_{l}x_{kl} = 1∀ k : ∑ l x k l = 1 and every LaTeX: x_{kl}\in\{0,1\}x k l ∈ { 0 , 1 } 
Compute the total similarity of matching: LaTeX: s^{\text{one2one}}\left(q,r\right)=\sum_{k,l}\:s_{kl}x_{kl}^*s one2one ( q , r ) = ∑ k , l s k l x k l ∗
For K=1, 2, 3, 4
Identify LaTeX: S_q^{\text{one2one}}(K)S q one2one ( K ) and LaTeX: S_q^{\text{many2many}}(K)S q many2many ( K ), i.e., the two sets of K most similar images from LaTeX: DD to LaTeX: qq with respect to the two matching criteria
Use the ground truth, given in image_retrieval.zip, to estimate: 
Precision: LaTeX: P^{\text{one2one}}\left(q,\:K\right)P one2one ( q , K ) and LaTeX: P^{\text{many2many}}\left(q,\:K\right)P many2many ( q , K )
Recall: LaTeX: R^{\text{one2one}}\left(q,\:K\right)R one2one ( q , K ) and LaTeX: R^{\text{many2many}}\left(q,\:K\right)R many2many ( q , K )
Use the ground truth, given in image_retrieval.zip, to compute your average precision and recall:
Precision:  LaTeX: P^{\text{one2one}}(K)=\frac{1}{|Q|}\sum_{q\in Q} P^{\text{one2one}}(q,K)P one2one ( K ) = 1 | Q | ∑ q ∈ Q P one2one ( q , K ) and LaTeX: P^{\text{many2many}}(K)=\frac{1}{|Q|}\sum_{q\in Q} P^{\text{many2many}}(q,K)P many2many ( K ) = 1 | Q | ∑ q ∈ Q P many2many ( q , K )
Recall: LaTeX: R^{\text{one2one}}(K)=\frac{1}{|Q|}\sum_{q\in Q} R^{\text{one2one}}(q,K)R one2one ( K ) = 1 | Q | ∑ q ∈ Q R one2one ( q , K ) and LaTeX: R^{\text{many2many}}(K)=\frac{1}{|Q|}\sum_{q\in Q} R^{\text{many2many}}(q,K)R many2many ( K ) = 1 | Q | ∑ q ∈ Q R many2many ( q , K ) 
 

What to Turn in?
A zipped folder that contains the following files:

(20  points) "precision_recall.pdf" document with a figure that shows two precision-recall curves of your image retrieval, where each curve uses the corresponding results: LaTeX: \{(P^{\text{one2one}}(K),R^{\text{one2one}}(K)):K=1,2,3,4\}{ ( P one2one ( K ) , R one2one ( K ) ) : K = 1 , 2 , 3 , 4 } andLaTeX: \{(P^{\text{many2many}}(K),R^{\text{many2many}}(K)):K=1,2,3,4\}{ ( P many2many ( K ) , R many2many ( K ) ) : K = 1 , 2 , 3 , 4 }. Plot the curves such that the precision values are on the y-axis and the recall values are on the x-axis.
(20) "retrieval.pth" Python file that has two matrices of your image retrieval, each with size 34 x 4, for the two matching formulations. In each matrix, the 34 rows correspond to the ID numbers of the query images, and the columns represent the ID numbers of the top K=4 most similar images retrieved from the dataset LaTeX: DD. 
(20) "features.pth" Python file that has a tensor of your deep features with size 170 x 20 x 128, for a total of 170 images where the first 34 are the query images and the remaining 136 are images from LaTeX: DD, and for the 20 keypoints detected in each image, and for the 128-dimensional deep features that you computed.
Submit your compressed folder on TEACH (Links to an external site.) before 8am on May 5, 2020.

Grading
60 points = If you submit the three files as described above.
+5 points = If your recall and precision are accurately computed from the two 34 x 4 matrices of your image retrieval in "retrieval.pth".
+5 points = If the two 34 x 4 matrices in "retrieval.pth" are accurately computed from your deep features in the 170 x 20 x 128 tensor stored in "features.pth".
+5 points = If your recall and precision for every K are worse by more than 2%  from ours.
+25 points = If your recall and precision for at least one value of K are slightly lower (within 2% margin) or better than ours.
