import cv2
import mediapipe as mp
import math
import numpy as np
import os
from ultralytics import YOLO
import pyautogui
import keyboard as kb
import threading
import time
from webcam import WebcamStream
from cal_point import *
from action_recorder import ActionRecorder

import subprocess
os.system("taskkill /IM blender.exe /F")
blender_path = "C:\\Program Files\\Blender Foundation\\Blender 3.4\\blender-launcher.exe"
blend_file = "ray_casting.blend"
python_script = "ray"

# 使用 subprocess 執行命令
command = f'"{blender_path}" {blend_file} --python-text {python_script}'
subprocess.run(command, shell=True)

# Initialize the hand landmark detector and yolo model
mp_hands = mp.solutions.hands.Hands()
yolo_model = YOLO("grab_0127.pt")

# initializing and starting multi-threaded webcam capture input stream 
FPS = 20
webcam_stream = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
webcam_stream.start()
# test a image
frame = webcam_stream.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
yolo_model(frame)
results = mp_hands.process(rgb_frame)
WIDTH, HEIGHT = frame.shape[1], frame.shape[0]

flag = False

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

yolo_speed = 0
yolo_ratio = 0.6
yolo_iou_thresh = 0.8
mp_speed = 0
skip_frame = 1
track_human_id = -1
loss_track_frame_count = 99
loss_track_frame_count_thresh = 10
operate = "none"
frame_rate = 0
act_list = ["point", "turn", "wave", "grab", "grab", "ok"]
del_act_rec = {"point": 0, "turn": 0, "wave": 0, "grab": 0, "ok": 0}
ok_cooldown = 0
show_giveup_option = False
zoom_in_out = 0

act_rec = ActionRecorder()
while True:
    mp_speed = 0
    # Capture the current frame
    key = webcam_stream.key
    frame = webcam_stream.read()
    
    if frame_rate % skip_frame == 0:
        t1 = time.time()
        yolo_result = yolo_model.track(frame, persist=True, conf=0.4, verbose=False)
        t2 = time.time()

        # Visualize the results on the frame
        if track_human_id == -1:
            annotated_frame = yolo_result[0].plot()
        else:
            if 6 in yolo_result[0].boxes.cls:
                data = yolo_result[0].boxes.data
                try:
                    tracking_data_idx = np.where(data[:, 4] == track_human_id)[0][0]
                    tracking_human_xyxy = data[tracking_data_idx][:4]
                    annotated_frame = cv2.rectangle(frame, (int(tracking_human_xyxy[0]), int(tracking_human_xyxy[1])), (int(tracking_human_xyxy[2]), int(tracking_human_xyxy[3])), (0, 255*act_rec.mouse_mode, 255), 2)
                except:
                    annotated_frame = frame
            if show_giveup_option == True: # show two rectangle to let user choose to give up tracking
                cv2.rectangle(annotated_frame, (0, HEIGHT//2), (WIDTH//2, HEIGHT), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (WIDTH//2, HEIGHT//2), (WIDTH, HEIGHT), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Give up", (0, HEIGHT//2 - 10), font, fontScale, fontColor, thickness, lineType)
                cv2.putText(annotated_frame, "Continue", (WIDTH//2, HEIGHT//2 - 10), font, fontScale, fontColor, thickness, lineType)

        # Display the annotated frame
        cv2.imshow("test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        yolo_speed = t2 - t1
        yolo_class = yolo_result[0].boxes.cls
        frames = []
        frame_rate = 0
        
        webcam_stream.set_cls(str(yolo_class))
        
        # can't find action/human
        if len(yolo_class) == 0 or yolo_result[0].boxes.is_track == False:
            continue
        
        # loss tracking human -> find new track_human_id
        if track_human_id <= -1:
            webcam_stream.set_cls("Loss human tracking")
            if 6 in yolo_class and 5 in yolo_class and ok_cooldown == 0:
                data = yolo_result[0].boxes.data
                idx1 = np.where(data[:, -1] == 6)[0]
                idx2 = np.where(data[:, -1] == 5)[0]
                # sort idx1 by bbox size
                idx1 = [idx1[idx_tmp] for idx_tmp in np.argsort((data[idx1][:, 2] - data[idx1][:, 0]) * (data[idx1][:, 3] - data[idx1][:, 1]))]
                # check all human bbox
                for i in idx1:
                    for j in idx2:
                        # print(get_overlap_ratio(data, i, j))
                        if get_overlap_ratio(data, i, j) > yolo_iou_thresh:
                            # success to track human and get id
                            track_human_id = yolo_result[0].boxes.id[i]
                            loss_track_frame_count = 0
                            webcam_stream.set_cls("start human tracking")
                            time.sleep(3)
                            break
        # if tracking id is not in the frame
        elif track_human_id not in yolo_result[0].boxes.id:
            # if there's still human in the frame and someone is doing action
            if 6 in yolo_class and len(np.where(yolo_class != 6)) > 0:
                loss_track_frame_count += 1
            if loss_track_frame_count >= loss_track_frame_count_thresh:
                # reset track_human_id
                track_human_id = -1
                show_giveup_option = False
                act_rec.mouse_mode = False
        # find human -> detect the action
        else:
            data = yolo_result[0].boxes.data
            tracking_data_idx = np.where(data[:, 4] == track_human_id)[0][0]
            highest_conf = 0
            act = []
            for idx, i in enumerate(data):
                if i[-1] == 6:
                    continue
                if act_rec.mouse_mode == True and i[-1] not in [0, 5]: # only point and ok can be detected in mouse mode
                    continue
                if get_overlap_ratio(data, tracking_data_idx, idx) > yolo_iou_thresh:
                    # find higest conf action
                    if i[5] > highest_conf:
                        highest_conf = i[5]
                        act = i
            if len(act) > 0:
                x1, y1, x2, y2 = extend_bbox(act[:4], extend_ratio=1.2, img_width=WIDTH, img_height=HEIGHT)
                rgb_frame = cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB)
                t1 = time.time()
                mp_results = mp_hands.process(rgb_frame)
                mp_speed = time.time() - t1
                # use mediapipe to check point action and direction
                if mp_results.multi_hand_landmarks:
                    for hand_landmarks in mp_results.multi_hand_landmarks:
                        act_rec.add_result(act, hand_landmarks.landmark)
                        del_act_rec[act_list[int(act[-1])]] = int(yolo_fps * 1.5 + 1)
                        break
            # If mouse_mode is True, check if the bounding box of the tracked human overlaps with the hand
            if act_rec.mouse_mode == True: 
                if len(act) > 0 and act[-1] != 5: # not doing ok gesture
                    x1, y1, x2, y2 = data[tracking_data_idx][:4]
                    rgb_frame = cv2.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2RGB)
                    t1 = time.time()
                    mp_results = mp_hands.process(rgb_frame)
                    mp_speed += time.time() - t1
                    if mp_results.multi_hand_landmarks:
                        for hand_landmarks in mp_results.multi_hand_landmarks:
                            hand_bbox = get_hand_bbox(hand_landmarks.landmark, WIDTH, HEIGHT)
                            if get_overlap_ratio(data, tracking_data_idx, 0, mp_data=hand_bbox) > yolo_iou_thresh:
                                # Record the hand keypoints information
                                act_rec.add_result([0, 0, 0, 0, 0], hand_landmarks.landmark)
                                del_act_rec["point"] = int(yolo_fps * 1.5 + 1)
            # delete action record after 1.5 seconds
            for k in del_act_rec:
                if del_act_rec[k] >= 0:
                    del_act_rec[k] -= 1
                else:
                    act_rec.yolo_results[k].clear()
                    act_rec.mediapipe_results[k].clear()
        
        if ok_cooldown > 0:
            ok_cooldown -= 1

        # calculate the frame rate
        yolo_fps = 1 / (yolo_speed + mp_speed)
        loss_track_frame_count_thresh = yolo_fps * 5  # wait 5 seconds to reset track_human_id
        skip_frame = int((FPS - yolo_fps) / yolo_fps)
        
    else:
        frames.append(frame)
    
    frame_rate += 1

    operate = "none"
    # decide which action to perform
    # 0:point  1:turn  2:wave  3:grab_in  4:grab_out  5:ok  6:human

    # point
    if len(act_rec.yolo_results["point"]) >= 1:
        l = act_rec.mediapipe_results["point"][-1]
        i = l[5]
        j = l[8]
        # write xyz to file and blender will read it
        with open("mpData.txt", "w") as f:
            print(j.x - i.x, i.y - j.y, i.z - j.z, file=f)
        operate = "point"
    # ok
    if len(act_rec.yolo_results["ok"]) >= 1 and ok_cooldown == 0:
        ok_mp_data = act_rec.mediapipe_results["ok"][-1]
        ok_yolo_data = act_rec.yolo_results["ok"][-1]
        # double check ok gesture by mediapipe
        if check_point_dist(ok_mp_data, 4, 8) <= 0.06 and check_point_dist(ok_mp_data, 4, 20) >= 0.17:
            if len(np.where(data[:, 4] == track_human_id)[0]) > 0:
                tracking_data_idx = np.where(data[:, 4] == track_human_id)[0][0]
                tracking_human_xyxy = data[tracking_data_idx][:4]
    
                # check if ok gesture is above the human bbox => give up tracking
                if act_rec.mouse_mode == False and tracking_human_xyxy[1] <= ok_yolo_data[1] + HEIGHT * 0.1: # ok gesture is at top of the human bbox
                    show_giveup_option = True
                elif show_giveup_option == True: # left to give up, right to continue
                    centerx, centery = (ok_yolo_data[0] + ok_yolo_data[2]) // 2, (ok_yolo_data[1] + ok_yolo_data[3]) // 2
                    if centerx <= WIDTH//2 and centery >= HEIGHT//2:
                        track_human_id = -1
                        show_giveup_option = False
                    elif centerx >= WIDTH//2 and centery >= HEIGHT//2:
                        show_giveup_option = False
                # normal ok
                elif act_rec.mouse_mode == True and len(act_rec.yolo_results["ok"]) >= yolo_fps * 2: # do action for 2 seconds
                    operate = "ok"
                
                act_rec.reset()
                ok_cooldown = int(yolo_fps * 3 + 1)
    # turn
    elif len(act_rec.yolo_results["turn"]) >= yolo_fps: # do action for 1 second
        act_cnt = 0
        turn_mp_data = act_rec.mediapipe_results["turn"]
        for i in range(len(turn_mp_data)-1):
            # thumb tip is going down and pinky tip is going up
            if turn_mp_data[i][4].y < turn_mp_data[i+1][4].y: #and turn_mp_data[i][20].y > turn_mp_data[i+1][20].y:
                act_cnt += 1
        if act_cnt >= yolo_fps*0.5:
            operate = "turn"
            act_rec.reset()
            del_act_rec["turn"] = 0
    # wave
    elif len(act_rec.yolo_results["wave"]) >= yolo_fps: # do action for 1 second
        act_cnt = 0
        act_left_right = 0
        wave_yolo_data = act_rec.yolo_results["wave"]
        # check if middle x of bbox is going left or right
        for i in range(len(wave_yolo_data)-1):
            if wave_yolo_data[i][0] + wave_yolo_data[i][2] <= wave_yolo_data[i+1][0] + wave_yolo_data[i+1][2]: # going right
                act_cnt += 1
                act_left_right += 1
            elif wave_yolo_data[i][0] + wave_yolo_data[i][2] > wave_yolo_data[i+1][0] + wave_yolo_data[i+1][2]: # going left
                act_cnt += 1
                act_left_right -= 1
        if act_cnt >= yolo_fps*1.3:
            if act_left_right > 0:
                operate = "wave_right"
            else:
                operate = "wave_left"
            act_rec.reset()
            del_act_rec["wave"] = 0
    # grab
    elif len(act_rec.yolo_results["grab"]) >= yolo_fps: # do action for 1 second
        act_cnt = 0
        grab_yolo_data = act_rec.yolo_results["grab"]

        # check grab in or out
        grab_in_out = "none"
        # check [in -> out] or [out -> in]
        middle_idx = len(grab_yolo_data) // 2
        front_avg = np.mean([tmp_data[-1] for tmp_data in grab_yolo_data[:middle_idx]])
        back_avg = np.mean([tmp_data[-1] for tmp_data in grab_yolo_data[middle_idx:]])
        if front_avg > back_avg: # [out -> in] (4 -> 3)
            grab_in_out = "out2in"
        elif front_avg == back_avg:
            grab_in_out = "none"
        else: # [in -> out] (3 -> 4)
            grab_in_out = "in2out"

        # check grab side or front
        max_x = np.max([tmp_data[0] + tmp_data[2] for tmp_data in grab_yolo_data])
        min_x = np.min([tmp_data[0] + tmp_data[2] for tmp_data in grab_yolo_data])
        if max_x / min_x > 1.3:
            grab_side = "side"
        else:
            grab_side = "front"

        if grab_in_out == "in2out":
            if grab_side == "side":
                operate = "grab_i2o_side"
            else:
                operate = "grab_i2o_front"
        else:
            if grab_side == "side":
                operate = "grab_o2i_side"
            else:
                operate = "grab_o2i_front"
        act_rec.reset()
        del_act_rec["grab"] = 0

    # perform operation
    if operate == "point":
        move_mouse_thresh_pixel = 50 / yolo_fps
        with open("final_point.txt", "r") as f:
            try:
                x, y = f.readline().split()
                x = int(x)
                y = int(y)
                x = max(1, x)
                x = min(WIDTH, x)
                y = max(1, y)
                y = min(HEIGHT, y)
                orix, oriy = pyautogui.position()
                if abs(x - orix) >= move_mouse_thresh_pixel or abs(y - oriy) >= move_mouse_thresh_pixel:
                    print(f"[Point]: Move mouse to ({x}, {y})")
                    lm = act_rec.mediapipe_results["point"][-1]
                    if check_point_dist(lm, 4, 10) / check_point_dist(lm, 17, 18) > 1 and 110 > check_vec_equ(lm, 2, 4, 13, 14) > 45:
                        pyautogui.mouseDown()
                    else:
                        pyautogui.mouseUp()
                    pyautogui.moveTo(x, y, duration=0.1)
            except:
                print("error")
        if act_rec.mouse_mode == False and len(act_rec.mediapipe_results["point"]) >= yolo_fps * 3:
            print("[Point]: Mouse mode activated")
            act_rec.mouse_mode = True
        if act_rec.mouse_mode == True:
            act_rec.reset()
        del_act_rec["point"] = 0
    if operate == "ok": # mouse mode deactivate
        if act_rec.mouse_mode == True:
            print("[Ok]: Mouse mode deactivated")
            act_rec.mouse_mode = False
    elif operate == "turn": # change mouse function
        # arrow/pen/eraser/laser
        act_rec.mouse_function_idx = (act_rec.mouse_function_idx + 1) % len(act_rec.mouse_function_list)
        current_function = act_rec.mouse_function_list[act_rec.mouse_function_idx]
        print(f"[Turn]: Change mouse function to {current_function}")
        if current_function == "arrow":
            kb.press_and_release('ctrl+a')
        elif current_function == "pen":
            kb.press_and_release('ctrl+p')
        elif current_function == "eraser":
            kb.press_and_release('ctrl+e')
        elif current_function == "laser":
            kb.press_and_release('ctrl+l')
    elif operate == "wave_right": # previous page
        print("[Wave]: Previous page")
        kb.press_and_release('left')
    elif operate == "wave_left": # next page
        print("[Wave]: Next page")
        kb.press_and_release('right')
    elif operate == "grab_i2o_side": #
        pass
    elif operate == "grab_o2i_side": #
        pass
    elif operate == "grab_i2o_front": # zoom in
        print("[Grab]: Zoom in")
        kb.press_and_release('ctrl+plus')
        zoom_in_out += 1
    elif operate == "grab_o2i_front": # zoom out
        if zoom_in_out >= 1:
            print("[Grab]: Zoom out")
            kb.press_and_release('ctrl+-')
            zoom_in_out -= 1
        else:
            print("[Grab]: Zoom out failed")
    else:
        pass

    if key == ord('q'):
        webcam_stream.stop()
        cv2.destroyAllWindows()
        os.system("taskkill /IM blender.exe /F")
        break