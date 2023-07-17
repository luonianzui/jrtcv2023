from module.ModuleDebugs import Debugs
from module.ModuleConfig import *


class Shared:
    @staticmethod
    # 获取主域锁状态----------------------------------------------------------------------------------------------------
    def Trys(in_conf, in_type):
        try:
            return in_conf[in_type]["LOCK"]
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Lock", 3)
                Debugs("Shared", "TRY/" + in_type, 3)
            return None

    @staticmethod
    # 加锁整个主域名----------------------------------------------------------------------------------------------------
    def Lock(in_conf, in_type, optputs=True):
        in_nums = 0
        try:
            while in_conf[in_type]["LOCK"]:
                in_nums = in_nums + 1
                if optputs and in_nums % 10000 == 0:
                    Debugs("Shared", "等待解锁: [" + in_type + "] 0x%02X" % (in_nums // 10000 % 255), 0)
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Lock", 3)
                Debugs("Shared", "LOC/" + in_type, 3)
            return False
        try:
            in_temp = in_conf[in_type]
            in_temp["LOCK"] = True
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "LOC/" + in_type, 3)
            return False
        return True

    @staticmethod
    # 解锁整个主域名----------------------------------------------------------------------------------------------------
    def Open(in_conf, in_type, optputs=True):
        try:
            if not in_conf[in_type]["LOCK"]:
                Debugs("Shared", "解锁失败: [" + in_type + "]", 3)
                return False
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Lock", 3)
                Debugs("Shared", "OPE/" + in_type, 3)
            return False
        try:
            in_temp = in_conf[in_type]
            in_temp["LOCK"] = False
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "OPE/" + in_type, 3)
            return False
        return True

    @staticmethod
    # 获取TTL时间，如果已达TTL则返回True，否则返回False并且自减TTL------------------------------------------------------
    def Ttl(in_conf, in_type, in_name):
        Shared.Lock(in_conf, in_type)
        try:
            in_temp = in_conf[in_type]
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "TTL/" + in_type, 3)
            Shared.Open(in_conf, in_type)
            return False
        try:
            if Shared.Ptr(in_conf, in_type, in_name) is None:
                in_retu = False
            else:
                if in_temp[in_name]['t'] == 0:
                    in_retu = True
                else:
                    in_retu = False
                    in_temp[in_name]['t'] = in_temp[in_name]['t'] - 1
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "TTL/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return False
        try:
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "OPE/" + in_type, 3)
            Shared.Open(in_conf, in_type)
            return False
        Shared.Open(in_conf, in_type)
        return in_retu

    @staticmethod
    # 返回所有当前主域下的子变量----------------------------------------------------------------------------------------
    def Lst(in_conf, in_type):
        Shared.Lock(in_conf, in_type)
        try:
            in_temp = in_conf[in_type]
            in_retu = []
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "LST/" + in_type, 3)
            Shared.Open(in_conf, in_type)
            return False
        try:
            for in_retu in in_temp:
                in_retu.append(in_temp[in_retu])
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Name", 3)
                Debugs("Shared", "LST/" + in_type, 3)
            pass
        try:
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "LST/" + in_type, 3)
            Shared.Open(in_conf, in_type)
            return False
        Shared.Open(in_conf, in_type)
        return in_retu

    @staticmethod
    # 添加新的变量值----------------------------------------------------------------------------------------------------
    def Add(in_conf, in_type, in_name, in_data, in_flag=True, optputs=True):
        Shared.Lock(in_conf, in_type)
        try:
            in_temp = in_conf[in_type]
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "ADD/" + in_type, 3)
            Shared.Open(in_conf, in_type)
            return False
        try:
            in_temp[in_name]
        except KeyError or BaseException:
            in_retu = True
            in_temp[in_name] = {
                'd': in_data,
                'c': in_flag,
            }
        else:
            in_retu = False
        try:
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
            Shared.Open(in_conf, in_type)
            return False
        Shared.Open(in_conf, in_type)
        return in_retu

    @staticmethod
    # 删除已有变量值----------------------------------------------------------------------------------------------------
    def Del(in_conf, in_type, in_name):
        Shared.Lock(in_conf, in_type)
        try:
            in_temp = in_conf[in_type]
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "DEL/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return False
        try:
            if Shared.Ptr(in_conf, in_type, in_name) is None:
                in_retu = False
            else:
                in_retu = True
                in_temp.pop(in_name)
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Name", 3)
                Debugs("Shared", "DEL/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return False
        try:
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "DEL/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return False
        Shared.Open(in_conf, in_type)
        return in_retu

    @staticmethod
    # 返回变量实体------------------------------------------------------------------------------------------------------
    def Ptr(in_conf, in_type, in_name):
        try:
            in_retu = in_conf[in_type][in_name]['c']
        except KeyError or BaseException:
            if shared_debug_flag and shared_detail_inf:
                Debugs("Shared", "读取失败：" + in_name, 1)
            in_retu = None
        return in_retu

    @staticmethod
    # 返回变量内容------------------------------------------------------------------------------------------------------
    def Get(in_conf, in_type, in_name, in_flag=True):
        Shared.Lock(in_conf, in_type)
        try:
            in_temp = in_conf[in_type]
            if Shared.Ptr(in_conf, in_type, in_name) is not None:
                in_retu = in_temp[in_name]['d']
                if in_flag:
                    in_temp[in_name]['c'] = False
            else:
                in_retu = None
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Unkw", 3)
                Debugs("Shared", "GET/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return None
        try:
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "GET/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return False
        Shared.Open(in_conf, in_type)
        return in_retu

    @staticmethod
    # 修改变量内容------------------------------------------------------------------------------------------------------
    def Put(in_conf, in_type, in_name, in_data, in_flag=True):
        Shared.Lock(in_conf, in_type)
        try:
            in_temp = in_conf[in_type]
            if Shared.Ptr(in_conf, in_type, in_name) is not None:
                in_temp[in_name]['d'] = in_data
                in_retu = True
                if in_flag:
                    in_temp[in_name]['c'] = True
            else:
                in_retu = False
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Unkw", 3)
                Debugs("Shared", "PUT/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return None
        try:
            in_conf[in_type] = in_temp
        except KeyError or BaseException:
            if shared_debug_flag:
                Debugs("Shared", "严重的问题:Type", 3)
                Debugs("Shared", "PUT/" + in_type + "/" + in_name, 3)
            Shared.Open(in_conf, in_type)
            return False
        Shared.Open(in_conf, in_type)
        return in_retu
