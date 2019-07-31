import os
import torch
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from yolo.preprocess import prep_image, prep_frame, inp_to_image
import cv2
import torch.utils.data
from opt import opt
from dataloader import  Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from yolo.darknet import Darknet
from pPose_nms import pose_nms, write_json
from utils_kzx import crop_from_dets
from sklearn.datasets import load_files
import time

if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame

def load_pose_model(args):
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    return pose_model

def load_yolo_model(args):
    print('loading yolo model ...')
    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights('models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = args.inp_dim
    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    det_model.cuda()
    det_model.eval()
    return det_model,det_inp_dim

class Pose():
    def __init__(self,args_):
        self.args=args_
        os.environ['CUDA_VISIBLE_DEVICES']=self.args.gpuid
        self.pose_model=load_pose_model(self.args)
        self.det_model,self.det_inp_dim=load_yolo_model(self.args)
    def get_pose(self,img_names):
        if len(img_names)>1:
            start_lc = 4000
            start_rc = 4000
            now_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            print('========START-Ten========')
            final_result=[]
            vis_images = []
            height_difference=[]
            for img_index in range(len(img_names)):
                print('--------------------')
                img_name=img_names[img_index]
                try:
                    img ,orig_img,im_name,im_dim_list= [],[],[],[]
                    inp_dim = int(self.args.inp_dim)
                    im_name_k = img_name
                    img_k, orig_img_k, im_dim_list_k = prep_image(im_name_k, inp_dim)
                    img.append(img_k)
                    orig_img.append(orig_img_k)
                    im_name.append(im_name_k)
                    im_dim_list.append(im_dim_list_k)
                except:
                    print('index-{}: image have problem'.format(img_index))
                    final_result.append((None,None))
                    continue
                with torch.no_grad():
                    img = torch.cat(img)
                    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

                    img = img.cuda()
                    prediction = self.det_model(img, CUDA=True)
                    dets = dynamic_write_results(prediction, self.args.confidence,
                                                 self.args.num_classes, nms=True, nms_conf=self.args.nms_thesh)
                    if isinstance(dets, int) or dets.shape[0] == 0:
                        print('index-{}: No person detected'.format(img_index))
                        final_result.append((None,None))
                        height_difference.append(None)
                        continue
                    dets = dets.cpu()
                    im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                    scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)
                    dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                    dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
                    dets[:, 1:5] /= scaling_factor
                    for j in range(dets.shape[0]):
                        dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                        dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                    boxes = dets[:, 1:5]
                    scores = dets[:, 5:6]
                    k=0
                    boxes_k = boxes[dets[:,0]==k]
                    inps = torch.zeros(boxes_k.size(0), 3, self.args.inputResH, self.args.inputResW)
                    pt1 = torch.zeros(boxes_k.size(0), 2)
                    pt2 = torch.zeros(boxes_k.size(0), 2)

                    orig_img, im_name, boxes, scores, inps, pt1, pt2=orig_img[k], im_name[k], boxes_k, scores[dets[:,0]==k], inps, pt1, pt2
                    inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                    inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                    batchSize = self.args.posebatch
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover

                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                        hm_j = self.pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    hm_data = hm.cpu()
                    orig_img = np.array(orig_img, dtype=np.uint8)
                    im_name=im_name.split('/')[-1]
                    preds_hm, preds_img, preds_scores = getPrediction(
                        hm_data, pt1, pt2, self.args.inputResH, self.args.inputResW, self.args.outputResH, self.args.outputResW)
                    result = pose_nms(
                        boxes, scores, preds_img, preds_scores)
                    result = {
                        'imgname': im_name,
                        'result': result
                    }
                    img = vis_frame(orig_img, result)
                    vis_images.append(img)
                    outpur_dir = os.path.join(self.args.outputpath, 'vis')
                    outpur_dir_raw=os.path.join(self.args.outputpath, 'raw')
                    if not os.path.exists(outpur_dir):
                        os.makedirs(outpur_dir)
                    if not os.path.exists(outpur_dir_raw):
                        os.makedirs(outpur_dir_raw)
                width=img.shape[1]
                keypoints=[res['keypoints'][0] for res in result['result']]
                distance=[xy[0] - width/2 for xy in keypoints]
                distance=torch.tensor([torch.abs(m) for m in distance])
                indice=torch.argsort(distance)[0]
                pose_result = result['result'][indice]['keypoints']
                # left_arm = pose_result[[6, 8, 10]].numpy()
                # right_arm = pose_result[[5, 7, 9]].numpy()
                # ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip',
                # 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
                left_arm = pose_result[[ 10]].numpy().astype(int)
                right_arm = pose_result[[ 9]].numpy().astype(int)
                left_arm_c_y=np.mean(left_arm,axis=0)[1]
                right_arm_c_y=np.mean(right_arm, axis=0)[1]
                # left_arm_c = tuple(np.mean(left_arm, axis=0).astype(int))
                # right_arm_c = tuple(np.mean(right_arm, axis=0).astype(int))
                left_arm_c=tuple(left_arm[0])
                right_arm_c=tuple(right_arm[0])
                hd=np.abs(left_arm_c_y-right_arm_c_y)
                height_difference.append(hd)

                cv2.circle(img, left_arm_c, 10, (0, 255, 0), -1, 8)
                cv2.circle(img, right_arm_c, 10, (0, 255, 0), -1, 8)
                log__vis_name=now_time+'-'+im_name
                cv2.imwrite(os.path.join(outpur_dir_raw, log__vis_name),orig_img)
                cv2.imwrite(os.path.join(outpur_dir, log__vis_name), img)
                if start_lc == 4000 and start_rc == 4000:
                    start_lc=left_arm_c_y
                    start_rc=right_arm_c_y
                    left_move=0
                    right_move=0
                else:
                    left_move=left_arm_c_y-start_lc
                    right_move=right_arm_c_y-start_rc
                print('index-{}--{}: left_c {:0f},right_c {:0f}'.format(img_index,im_name, left_arm_c_y, right_arm_c_y))
                print('index-{}--{}: start_lc {:0f},start_rc {:0f}'.format(img_index,im_name, start_lc, start_rc))
                print('index-{}--{}: left_move {:0f},right_move {:0f}'.format(img_index,im_name, left_move, right_move))
                print('index-{}--{}: height_difference {:0f}'.format(img_index, im_name, hd))
                final_result.append((left_move,right_move))
            return final_result,vis_images,now_time,height_difference

        elif len(img_names)==1:
            now_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            print('========START-One========')
            final_result = []
            vis_images = []
            height_difference=[]
            for img_index in range(len(img_names)):
                img_name = img_names[img_index]
                try:
                    img, orig_img, im_name, im_dim_list = [], [], [], []
                    inp_dim = int(self.args.inp_dim)
                    im_name_k = img_name
                    img_k, orig_img_k, im_dim_list_k = prep_image(im_name_k, inp_dim)
                    img.append(img_k)
                    orig_img.append(orig_img_k)
                    im_name.append(im_name_k)
                    im_dim_list.append(im_dim_list_k)
                except:
                    print('index-{}: image have problem'.format(img_index))
                    final_result.append((None, None))
                with torch.no_grad():
                    img = torch.cat(img)
                    vis_img=img.numpy()[0]
                    vis_img=np.transpose(vis_img, (1, 2, 0))
                    vis_img=vis_img[:,:,::-1]
                    vis_images.append(vis_img)
                    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                    img = img.cuda()
                    prediction = self.det_model(img, CUDA=True)
                    dets = dynamic_write_results(prediction, self.args.confidence,
                                                 self.args.num_classes, nms=True, nms_conf=self.args.nms_thesh)
                    if isinstance(dets, int) or dets.shape[0] == 0:
                        print('index-{}: No person detected'.format(img_index))
                        final_result.append((None, None))
                    else:
                        print('index-{}: Person detected'.format(img_index))
                        final_result.append((4,4))
            return final_result,vis_images,now_time,height_difference

    def predict(self,img_names):
        results,vis_imgs,now_time,height_difference=self.get_pose(img_names)
        print('==========ANALYSISING==========')
        if len(results)>1:
            print('Moves:',results)
            print('Difference:',height_difference)
            prediction=[]
            for moves in results:
                if None in moves:
                    prediction.append(2)
                elif moves[0]<=60 and moves[1]<=60:
                    prediction.append(0)
                else:prediction.append(1)
            print('Move Prediction:',prediction)

            prediction_hd=[]
            for hd in height_difference:
                if hd==None:
                    prediction_hd.append(2)
                elif hd<80:
                    prediction_hd.append(0)
                else:
                    prediction_hd.append(1)
            print('Height Prediction:',prediction_hd)

            if prediction.count(2)>=3 or prediction_hd.count(2)>=3:
                print('There is no person in at least 3 images!!!')
                result=int(0)
            elif prediction.count(1)>3 or prediction_hd.count(1)>3:
                print('There maybe some problems with arms!!!')
                result=int(1)
            else:
                print('There is no problem with arms.')
                result=int(2)
            print('Result==========>',result)
            with open('log/result.txt','a') as f:
                f.write('=====================\n')
                f.write('Time:{}\n'.format(now_time))
                f.write('Moves:{}\n'.format(results))
                f.write('Prediction:{}\n'.format(prediction))
                f.write('Height:{}\n'.format(height_difference))
                f.write(('Predicted:{}\n').format(prediction_hd))
                f.write('Result:{}\n'.format(result))
            return result,vis_imgs
        elif len(results)==1:
            if None in results[0]:
                result=int(5)
            else:
                result=int(4)
            print('Result==========>', result)
            return result,vis_imgs

def read_files(path):
    files=[]
    for img in os.listdir(path):
        files.append(os.path.join(path,img))
    files.sort()
    return files

Pose_model=Pose(opt)
images=read_files('/home/kkkzxx/Desktop/pose/4')
r,ims=Pose_model.predict(images)
cv2.imshow('img',ims[0])
cv2.waitKey()