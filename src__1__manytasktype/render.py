from tkinter import *
import random
import numpy as np


def dispatch_color(edge_color , E):
    """
    分配颜色，edge_color列表和 E 列表对应

    :param edge_color: 颜色列表 空的
    :param E: 设备列表
    :return: 颜色列表
    """
    for egde_id in range(len(E)):
        color = '#' + str("%03d" % random.randint(0, 255))[2:] + str("%03d" % random.randint(0, 255))[2:] + str("%03d" %random.randint(0, 255))[2:]
        edge_color.append(color)
    return edge_color

def get_info(U, MAX_EP_STEPS):
    """
    取得所有用户对应文件的第二列 也就是user.mob的第一列 然后找到他们中的最小值和最大值  作为x轴
    取得所有用户对应文件的第三列(前100行) 也就是user.mob的第二列(前3000行) 然后找到他们中的最小值和最大值  作为y轴

    :param U: 用户列表
    :param MAX_EP_STEPS: 每一回合的最大步数 3000
    :return: 返回 x_min, x_Max, y_min, y_Max
    """
    x_min, x_Max, y_min, y_Max = np.inf, -np.inf, np.inf, -np.inf
    # 取得所有用户对应文件的第二列 也就是user.mob的第一列 然后找到他们中的最小值和最大值  作为x轴
    for user in U:
       if(max(user.mob[:, 0]) > x_Max):  # user.mob 是这个用户对应文件每行(都*30后)的第二列和第三列的矩阵
           x_Max = max(user.mob[:, 0])
       if(min(user.mob[:, 0]) < x_min):
           x_min = min(user.mob[:, 0])
    # y axis
    for user in U:
        if (max(user.mob[:MAX_EP_STEPS, 1]) > y_Max):
            y_Max = max(user.mob[:MAX_EP_STEPS, 1])
        if (min(user.mob[:MAX_EP_STEPS, 1]) < y_min):
            y_min = min(user.mob[:MAX_EP_STEPS, 1])
    return x_min, x_Max, y_min, y_Max

#####################  hyper parameters  ####################
MAX_SCREEN_SIZE = 1000  # 弹出框大小
EDGE_SIZE = 35  # 设备点的大小
USER_SIZE = 15  # 用户点的大小

#####################  User  ####################
class oval_User:
    def __init__(self, canvas, color, user_id):
        self.user_id = user_id
        self.canvas = canvas
        # 根据位置和颜色创建圆形
        self.id = canvas.create_oval(500, 500, 500 + USER_SIZE, 500 + USER_SIZE, fill=color)

    def draw(self, vector, edge_color, user):  # （移动向量，此用户对应的ES的颜色，此用户对象）
        info = self.canvas.coords(self.id)  # 提取位置信息
        self.canvas.delete(self.id)  # 提取位置信息后删除之前创建的圆形
        # connect
        if user.req.state != 5 and user.req.state != 6:
            self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill=edge_color)  # 根据提取到的位置信息创建圆形
        # not connected
        else:
            # disconnection
            if user.req.state == 5:
                self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill="red")
            # migration
            elif user.req.state == 6:
                self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill="green")
        # move the user
        self.canvas.move(self.id, vector[0][0], vector[0][1])  # 移动圆形，vector[0][0]为x的方向，vector[0][1]为y的方向

#####################  Edge  ####################
class oval_Edge:
    def __init__(self, canvas, color, edge_id):
        self.edge_id = edge_id
        self.canvas = canvas
        self.id = canvas.create_oval(500, 500, 500 + EDGE_SIZE, 500 + EDGE_SIZE, fill=color)

    def draw(self, vector):
        """
        根据移动向量去移动椭圆

        :param vector: ES移动向量
        """
        self.canvas.move(self.id, vector[0][0], vector[0][1])

#####################  convas  ####################
class Demo:
    def __init__(self, E, U, O, MAX_EP_STEPS):
        # create canvas
        self.x_min, self.x_Max, self.y_min, self.y_Max = get_info(U, MAX_EP_STEPS)
        self.tk = Tk()  # 容器
        self.tk.title("Simulation: Resource Allocation in Egde Computing Environment")
        self.tk.resizable(0, 0)  # tk大小不可调节
        self.tk.wm_attributes("-topmost", 0)  # 0 ：tk不会显示到最上面
        # Canvas组件
        self.canvas = Canvas(self.tk, width=MAX_SCREEN_SIZE+500, height=1000, bd=0, highlightthickness=0, bg='black')
        self.canvas.pack()
        self.tk.update()
        # 根据图形在坐标轴上的范围和当前屏幕大小，计算出正确的缩放比例，以便将图形缩放到适当的大小并显示在屏幕上
        x_range = self.x_Max - self.x_min  # x轴
        y_range = self.y_Max - self.y_min  # y轴
        self.rate = x_range/y_range
        if self.rate > 1:
            self.x_rate = (MAX_SCREEN_SIZE / x_range)
            self.y_rate = (MAX_SCREEN_SIZE / y_range) * (1/self.rate)
        else:
            self.x_rate = (MAX_SCREEN_SIZE / x_range) * (self.rate)
            self.y_rate = (MAX_SCREEN_SIZE / y_range)

        self.edge_color = []
        self.edge_color = dispatch_color(self.edge_color, E)  # len(edge_color) = 10
        self.oval_U, self.oval_E = [], []
        # initialize the object
        for edge_id in range(len(E)):
            self.oval_E.append(oval_Edge(self.canvas, self.edge_color[edge_id], edge_id))
        for user_id in range(len(U)):  # 这个用户卸载到哪一个ES，就用哪一个ES对应的颜色
            self.oval_U.append(oval_User(self.canvas, self.edge_color[int(O[user_id])], user_id))

    def draw(self, E, U, O):
        """

        :param E: ES列表
        :param U: 用户列表
        :param O: 用户卸载列表 [9，9，9，2，9，0，3，4，4，3] 一个用户对应一个ES
        """
        # edge
        edge_vector = np.zeros((1, 2))
        for edge in E:
            edge_vector[0][0] = (edge.loc[0] - self.x_min) * self.x_rate - self.canvas.coords(self.oval_E[edge.edge_id].id)[0]
            edge_vector[0][1] = (edge.loc[1] - self.y_min) * self.y_rate - self.canvas.coords(self.oval_E[edge.edge_id].id)[1]
            self.oval_E[edge.edge_id].draw(edge_vector)  # edge_vector ：移动位置
        # user
        user_vector = np.zeros((1, 2))
        for user in U:
            user_vector[0][0] = (user.loc[0][0] - self.x_min) * self.x_rate - self.canvas.coords(self.oval_U[user.user_id].id)[0]
            user_vector[0][1] = (user.loc[0][1] - self.y_min) * self.y_rate - self.canvas.coords(self.oval_U[user.user_id].id)[1]
            self.oval_U[user.user_id].draw(user_vector, self.edge_color[int(O[user.user_id])], user)  # self.edge_color[int(O[user.user_id])] 此用户对应的ES颜色
        # 快速刷新屏幕
        self.tk.update_idletasks()
        self.tk.update()

#####################  Outer parameter  ####################
class UE():
    def __init__(self, user_id, data_num):
        self.user_id = user_id  # number of the user
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = "KAIST" + "_30sec_" + data_num + ".txt"  # LOCATION + "_30sec_" + data_num + ".txt"
        file_path = "../data/" + "KAIST/" + file_name  # LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0
        for line in f1:
            data += 1
        self.num_step = data * 30
        self.mob = np.zeros((self.num_step, 2))

        # write data to self.mob
        now_sec = 0
        for line in f1:
            for sec in range(30):
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30

    def mobility_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]   # x

        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf

class EdgeServer():
    def __init__(self, edge_id, loc):
        self.edge_id = edge_id  # edge server number
        self.loc = loc


