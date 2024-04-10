class ActionRecorder:
    def __init__(self):
        self.yolo_results = {"point": [], "turn": [], "wave": [], "grab": [], "ok": []}
        self.mediapipe_results = {"point": [], "turn": [], "wave": [], "grab": [], "ok": []}
        self.mouse_function_list = ["arrow", "pen", "eraser", "laser"]
        self.mouse_function_idx = 0
        self.mouse_mode = False

    def add_result(self, yolo_data, mp_data):
        yolo_cls = yolo_data[-1]
        if yolo_cls == 0:
            self.yolo_results["point"].append(yolo_data)
            self.mediapipe_results["point"].append(mp_data)
        elif yolo_cls == 1:
            self.yolo_results["turn"].append(yolo_data)
            self.mediapipe_results["turn"].append(mp_data)
        elif yolo_cls == 2:
            self.yolo_results["wave"].append(yolo_data)
            self.mediapipe_results["wave"].append(mp_data)
        elif yolo_cls == 3 or yolo_cls == 4:
            self.yolo_results["grab"].append(yolo_data)
            self.mediapipe_results["grab"].append(mp_data)
        elif yolo_cls == 5:
            self.yolo_results["ok"].append(yolo_data)
            self.mediapipe_results["ok"].append(mp_data)


    def reset(self):
        self.yolo_results = {"point": [], "turn": [], "wave": [], "grab": [], "ok": []}
        self.mediapipe_results = {"point": [], "turn": [], "wave": [], "grab": [], "ok": []}