# ----------------------------------------------
#
#           SCURM_Vision_Module_Camera
#               Coding By Pikachu
#                  摄像头API
#
#            LAST_UPDATE:OCT30/2020
#
# ----------------------------------------------
from module import ModuleMvsdks as mvsdks, ModuleDebugs as debugs
from module import ModuleConfig as config
import cv2
import time
import platform
import numpy as np

debug_head = "Camera"


class Camera:
    @staticmethod
    def list(in_flags=True):
        camera_list_devList = mvsdks.CameraEnumerateDevice()
        camera_list_devNums = len(camera_list_devList)
        if camera_list_devNums > 0:
            debugs.Debugs(debug_head, "连接相机数量: " + str(camera_list_devNums))
        else:
            debugs.Debugs(debug_head, "没有摄像机信息!", in_level=3)
            return 0
        for i, ModuleCamera_DevInfo in enumerate(camera_list_devList):
            if in_flags:
                debugs.Debugs(debug_head, "{}-{} {}".format(i, ModuleCamera_DevInfo.GetPortType()
                                                            , ModuleCamera_DevInfo.GetFriendlyName()[:7], ))
                debugs.Debugs(debug_head, "{}{}".format(ModuleCamera_DevInfo.GetFriendlyName()[10:-4],
                                                        ModuleCamera_DevInfo.GetFriendlyName()[27:]))

        return camera_list_devNums, camera_list_devList

    @staticmethod
    def PrintCapbility(cap):
        for i in range(cap.iTriggerDesc):
            desc = cap.pTriggerDesc[i]
            debugs.Debugs(debug_head, "拍摄方式：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iImageSizeDesc):
            desc = cap.pImageSizeDesc[i]
            debugs.Debugs(debug_head, "输出大小：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iClrTempDesc):
            desc = cap.pClrTempDesc[i]
            debugs.Debugs(debug_head, "捕获色温：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iMediaTypeDesc):
            desc = cap.pMediaTypeDesc[i]
            debugs.Debugs(debug_head, "输出位深：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iFrameSpeedDesc):
            desc = cap.pFrameSpeedDesc[i]
            debugs.Debugs(debug_head, "快门速度：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iPackLenDesc):
            desc = cap.pPackLenDesc[i]
            debugs.Debugs(debug_head, "传输包长：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iPresetLut):
            desc = cap.pPresetLutDesc[i]
            debugs.Debugs(debug_head, "捕获方法：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iAeAlmSwDesc):
            desc = cap.pAeAlmSwDesc[i]
            debugs.Debugs(debug_head, "曝光算法：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iAeAlmHdDesc):
            desc = cap.pAeAlmHdDesc[i]
            debugs.Debugs(debug_head, "转换个数：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iBayerDecAlmSwDesc):
            desc = cap.pBayerDecAlmSwDesc[i]
            debugs.Debugs(debug_head, "预设滤镜：{} - {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iBayerDecAlmHdDesc):
            desc = cap.pBayerDecAlmHdDesc[i]
            debugs.Debugs(debug_head, "输出质量：{} - {}".format(desc.iIndex, desc.GetDescription()))

    def __init__(self, in_nums=0):
        try:
            camera_init_nums, camera_init_list = Camera.list()
        except TypeError:
            debugs.Debugs(debug_head, "找不到相机设备!", 3)
            exit(0)
        if in_nums <= camera_init_nums - 1:
            try:
                # 初始化相机的设备
                self.camera = mvsdks.CameraInit(camera_init_list[in_nums], -1, -1)
                # 获取相机特性描述
                camera_init_caps = mvsdks.CameraGetCapability(self.camera)
                Camera.PrintCapbility(camera_init_caps)
                # 判断黑白彩色相机
                camera_init_mono = (camera_init_caps.sIspCapacity.bMonoSensor != 0)
                # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
                if camera_init_mono:
                    mvsdks.CameraSetIspOutFormat(self.camera, mvsdks.CAMERA_MEDIA_TYPE_MONO8)
                # 相机模式切换成连续采集
                mvsdks.CameraSetTriggerMode(self.camera, 0)
                # 手动设置曝光时间
                mvsdks.CameraSetAeState(self.camera, 0)
                mvsdks.CameraSetExposureTime(self.camera, (1000 // config.camera_resol_sfps) * 1000)
                # 让SDK内部取图线程开始工作
                mvsdks.CameraPlay(self.camera)
                # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
                camera_init_size = camera_init_caps.sResolutionRange.iWidthMax * camera_init_caps.sResolutionRange.iHeightMax * (
                    1 if camera_init_mono else 3)
                # 分配RGB buffer，用来存放ISP输出的图像
                # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
                self.buffer = mvsdks.CameraAlignMalloc(camera_init_size, 16)
            except mvsdks.CameraException as e:
                debugs.Debugs(debug_head, "CameraInit Failed:{}.format(e.error_code, e.message)", in_level=2)
                return
        else:
            pass

    def coff(self):
        debugs.Debugs(debug_head, "其他参数：", in_lines="")
        mvsdks.CameraUnInit(self.camera)
        mvsdks.CameraAlignFree(self.buffer)

    # 获取一帧图像
    def getf(self):
        try:
            camera_geti_pRawDatas, camera_geti_FrameHead = mvsdks.CameraGetImageBuffer(self.camera, 2000)
            mvsdks.CameraImageProcess(self.camera, camera_geti_pRawDatas, self.buffer, camera_geti_FrameHead)
            mvsdks.CameraReleaseImageBuffer(self.camera, camera_geti_pRawDatas)
            return self.buffer, camera_geti_FrameHead, self.camera
        except mvsdks.CameraException as e:
            debugs.Debugs(debug_head, "CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    def getp(self, widt=1280, high=1024):
        camera_getf_buff, camera_getf_head, camera_getf_coff = self.getf()
        if platform.system() == "Windows":
            mvsdks.CameraFlipFrameBuffer(camera_getf_buff, camera_getf_head, 1)
        # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
        # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
        camera_getf_data = (mvsdks.c_ubyte * camera_getf_head.uBytes).from_address(camera_getf_buff)
        camera_getf_oupt = np.frombuffer(camera_getf_data, dtype=np.uint8)
        camera_getf_oupt = camera_getf_oupt.reshape(
            (camera_getf_head.iHeight, camera_getf_head.iWidth,
             1 if camera_getf_head.uiMediaType == mvsdks.CAMERA_MEDIA_TYPE_MONO8 else 3))
        if config.camera_resol_tbgr:
            camera_getf_oupt = camera_getf_oupt[..., ::-1]
        camera_getf_oupt = cv2.resize(camera_getf_oupt, (config.camera_resol_widt, config.camera_resol_high),
                                      interpolation=cv2.INTER_LINEAR)
        return camera_getf_oupt

    def save(self, in_buff="", in_head="",
             in_path="./save_picture_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".bmp"):
        if in_buff == "":
            camera_save_buff, camera_save_head, camera_save_came = self.getf()
        else:
            camera_save_buff = in_buff
            camera_save_head = in_head
        # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
        # 该示例中我们只是把图片保存到硬盘文件中
        status = mvsdks.CameraSaveImage(self.camera, in_path, camera_save_buff, camera_save_head, mvsdks.FILE_BMP,
                                        100)
        if status == mvsdks.CAMERA_STATUS_SUCCESS:
            debugs.Debugs(debug_head, "Save image successfully")
            debugs.Debugs(debug_head, "image_size = {}X{}".format(camera_save_head.iWidth,
                                                                  camera_save_head.iHeight))
        else:
            debugs.Debugs(debug_head, "Save image failed. err={}".format(status))
