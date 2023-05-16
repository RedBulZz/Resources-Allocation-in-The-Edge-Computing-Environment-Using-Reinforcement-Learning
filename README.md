# ***Resources Allocation in The Edge Computing Environment Using Reinforcement Learning***
# ***使用强化学习在边缘计算环境中进行资源分配***

## Summary
The cloud computing based mobile applications, such as augmented reality (AR), face recognition, and object recognition have become popular in recent years. However, cloud computing may cause high latency and increase the backhaul bandwidth consumption because of the remote execution. To address these problems, edge computing can improve response times and relieve the backhaul pressure by moving the storage and computing resources closer to mobile users.
## 概括
近年来，基于云计算的移动应用，如增强现实（AR）、人脸识别和物体识别等已经开始流行。 然而，由于远程执行，云计算可能会导致高延迟并增加回程带宽消耗。 为了解决这些问题，边缘计算可以通过将存储和计算资源移动到更靠近移动用户的位置来缩短响应时间并减轻回程压力。

Considering the computational resources, migration bandwidth, and offloading target in an edge computing environment, the project aims to use Deep Deterministic Policy Gradient (DDPG), a kind of Reinforcement Learning (RL) approach, to allocate resources for mobile users in an edge computing environment.
考虑到边缘计算环境中的计算资源、迁移带宽和卸载目标，该项目旨在使用深度确定性策略梯度（DDPG），一种强化学习（RL）方法，为边缘计算中的移动用户分配资源 环境。

 ![gui](image/Summary.png)
 picture originated from: [IEEE Inovation at Work](https://innovationatwork.ieee.org/real-life-edge-computing-use-cases/)
***

## Prerequisite

+ Python 3.7.5
+ Tensorflow 2.2.0
+ Tkinter 8.6

***

## Build Setup
## 构建设置

### *Run The System*
### *运行系统*
```cmd
$ python3 src/run_this.py
```

### *Text Interface Eable / Diable* (in run_this.py)
### *文本界面启用/禁用*（在 run_this.py 中）
```python
TEXT_RENDER = True / False
```

### *Graphic Interface Eable / Diable* (in run_this.py)
### *图形界面启用/禁用*（在 run_this.py 中）
```python
SCREEN_RENDER = True / False
```

***

## Key Point
## 关键


## *Edge Computing Environment*
## *边缘计算环境*

+ Mobile User
  + Users move according to the mobility data provided by [CRAWDAD](https://crawdad.org/index.html). This data was collected from the users of mobile devices at the subway station in Seoul, Korea.
  用户根据[CRAWDAD](https://crawdad.org/index.html)提供的移动数据移动。 这些数据是从韩国首尔地铁站的移动设备用户那里收集的。
  + Users' devices offload tasks to one edge server to obtain computation service.
  用户的设备将任务卸载到一个边缘服务器以获得计算服务。
  + After a request task has been processed, users need to receive the processed task from the edge server and offload a new task to an edge server again.
  请求任务处理完成后，用户需要从边缘服务器接收处理后的任务，并再次卸载新的任务给边缘服务器。
+ Edge Server
  + Responsible for offering computational resources *(6.3 * 1e7 byte/sec)* and processing tasks for mobile users.
  负责为移动用户提供计算资源*（6.3 * 1e7 字节/秒）*和处理任务。
  + Each edge server can only provide service to limited numbers of users and allocate computational resources to them.
  每个边缘服务器只能为有限数量的用户提供服务，并为他们分配计算资源。
  + The task may be migrated from one edge server to another within limited bandwidth *(1e9 byte/sec)*.
  任务可以在有限带宽*（1e9 字节/秒）* 内从一个边缘服务器迁移到另一个边缘服务器。

+ Request Task: [VOC SSD300 Objection Detection](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)
  + state 1 : start to offload a task to the edge server 
  开始将任务卸载到边缘服务器
  + state 2 : request task is on the way to the edge server *(2.7 * 1e4 byte)*
  请求任务正在去往边缘服务器的路上 *(2.7 * 1e4 字节)*
  + state 3 : request task is proccessed *(1.08 * 1e6 byte)*
  请求任务已处理 *(1.08 * 1e6 字节)*
  + state 4 : request task is on the way back to the mobile user *(96 byte)*
  请求任务正在返回移动用户的路上 *（96 字节）*
  + state 5 : disconnect (default)
  断开连接（默认）
  + state 6 : request task is migrated to another edge server
  请求任务迁移到另一台边缘服务器

+ Graphic Interface 图形界面

  ![gui](image/gi.png)
  + Edge servers *(static)*
    + Big dots with consistent color
    颜色一致的大点
  + Mobile users *(dynamic)*
    + Small dots with changing color
    变色小点
    + Color
      + Red : request task is in state 5
      请求任务处于状态 5
      + Green : request task is in state 6
      请求任务处于状态 6
      + others : request task is handled by the edge server with the same color and is in state 1 ~ state 4
      请求任务由相同颜色的边缘服务器处理，处于状态1~状态4

## *Deep Deterministic Policy Gradient* (in DDPG.py)
## *深度确定性策略梯度*（在 DDPG.py 中）

+ Description
  
  While determining the offloading server of each user is a discrete variable problem, allocating computing resources and migration bandwidth are continuous variable problems. Thus, Deep Deterministic Policy Gradient (DDPG), a model-free off-policy actor-critic algorithm, can solve both discrete and continuous problems. Also, DDPG updates model weights every step, which means the model can adapt to a dynamic environment instantly.
  确定每个用户的卸载服务器是离散变量问题，分配计算资源和迁移带宽是连续变量问题。 因此，深度确定性策略梯度 (DDPG) 是一种无模型的 off-policy actor-critic 算法，可以解决离散和连续问题。 此外，DDPG 每一步都会更新模型权重，这意味着模型可以立即适应动态环境。
+ State

  ```python
    def generate_state(two_table, U, E, x_min, y_min):
        one_table = two_to_one(two_table)
        S = np.zeros((len(E) + one_table.size + len(U) + len(U)*2))
        count = 0
        for edge in E:
            S[count] = edge.capability/(r_bound*10)
            count += 1
        for i in range(len(one_table)):
            S[count] = one_table[i]/(b_bound*10)
            count += 1
        for user in U:
            S[count] = user.req.edge_id/100
            count += 1
        for user in U:
            S[count] = (user.loc[0][0] + abs(x_min))/1e5
            S[count+1] = (user.loc[0][1] + abs(y_min))/1e5
            count += 2
        return S
  ```

  + **Available computing resources** of each edge server
  + **每个边缘服务器的可用计算资源**
  + **Available migration bandwidth** of each connection between edge servers
  + **边缘服务器之间每个连接的可用迁移带宽**
  + **Offloading target** of each mobile user
  + **每个移动用户的卸载目标**
  + **Location** of each mobile user
  + **每个移动用户的位置**
   
   
   

+ Action

  ```python
  def generate_action(R, B, O):
    a = np.zeros(USER_NUM + USER_NUM + EDGE_NUM * USER_NUM)
    a[:USER_NUM] = R / r_bound
    # bandwidth
    a[USER_NUM:USER_NUM + USER_NUM] = B / b_bound
    # offload
    base = USER_NUM + USER_NUM
    for user_id in range(USER_NUM):
        a[base + int(O[user_id])] = 1
        base += EDGE_NUM
    return a
  ```

  + **Computing resources**  of each mobile user's task need to uses(continuous)
  + 每个移动用户的任务需要使用的**计算资源**（连续）
  + **Migration bandwidth** of each mobile user's task needs to occupy (continuous)
  + **每个移动用户的任务需要占用的迁移带宽**（连续）
  + **Offloading target** of each mobile user (discrete)
  + 每个移动用户的**卸载目标**（离散）

+ Reward
  + **Total processed tasks** in each step
  + **每个步骤中处理的任务总数**

+ Model Architecture  模型架构

  ![ddpg architecture](image/DDPG_architecture.png)

***

## Simulation Result  仿真结果

+ Simulation Environment
  + 模拟环境
  + 10 edge servers with computational resources *6.3 * 1e7 byte/sec*
  + 10 个具有计算资源的边缘服务器 *6.3 * 1e7 字节/秒*
  + Each edge server can provide at most 4 task processing services.
  + 每个边缘服务器最多可以提供4个任务处理服务。
  + 3000 steps/episode, 90000 sec/episode
  + 3000 步/回合，90000 秒/回合

+ Result
    | Number of Clients | Average Total proccessed tasks in the last 10 episodes| Training History |
    | 客户数量 | 最近 10 集中平均处理的任务总数| 培训历史 |
    | :-------: | :--------: | :--------: |
    | 10 | 11910 | ![result](output/ddpg_10u10e4lKAIST/rewards.png) |
    | 20 | 23449 | ![result](output/ddpg_20u10e4lKAIST/rewards.png) |
    | 30 | 33257 | ![result](output/ddpg_30u10e4lKAIST/rewards.png) |
    | 40 | 40584 | ![result](output/ddpg_40u10e4lKAIST/rewards.png) |

***

## Demo

+ Demo Environment
+ 演示环境
  + 35 mobile users and 10 edge servers in the environment
  + 环境中有 35 个移动用户和 10 个边缘服务器
  + Each edge server can provide at most 4 task processing services.
  + 每个边缘服务器最多可以提供4个任务处理服务。
  
+ Demo Video
  ![demo video](image/dm.mov)

***

## Reference  参考

+ Mobility Data
    移动数据
  [Mongnam Han, Youngseok Lee, Sue B. Moon, Keon Jang, Dooyoung Lee, CRAWDAD dataset kaist/wibro (v. 2008‑06‑04), downloaded from https://crawdad.org/kaist/wibro/20080604, https://doi.org/10.15783/C72S3B, Jun 2008.](https://crawdad.org/kaist/wibro/20080604)
