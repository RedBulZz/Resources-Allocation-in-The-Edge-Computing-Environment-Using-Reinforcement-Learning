from env import Env
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime



#####################  hyper parameters  ####################
CHECK_EPISODE = 4
LEARNING_MAX_EPISODE = 100
MAX_EP_STEPS = 3000  # 每一回合的最大步骤
TEXT_RENDER = False  # 文本界面启用/禁用
SCREEN_RENDER = True  # 图形界面启用/禁用
CHANGE = False
# SLEEP_TIME = 0.1  # 延迟时间

#####################  function  ####################
def exploration (a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a

###############################  training  ####################################

if __name__ == "__main__":
    print(datetime.datetime.now())
    env = Env()
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location = env.get_inf()
    # r_bound:ES最大资源(性能)1e9 * 0.063，b_boundL:ES之间最大带宽1e9
    ddpg = DDPG(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    b_var = 1
    episode_reward = []  # 记录每个回合的总奖励
    r_v, b_v = [], []
    var_reward = []
    max_rewards = 0
    episode = 0  # 记录回合数（不会重置）
    var_counter = 0  # 回合数（用于循环，可能重置）
    epoch_inf = []  # 记录每一回合的打印信息
    while var_counter < LEARNING_MAX_EPISODE:
        # initialize
        s = env.reset()
        episode_reward.append(0)  # 初始化 让每一回合的奖励能够累加

        if SCREEN_RENDER:
            env.initial_screen_demo()

        for j in range(MAX_EP_STEPS):  # 1回合3000步，也就是3000次向前传播，每个用户移动100次
            # time.sleep(SLEEP_TIME)
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()

            # DDPG
            # a 中的 offloading 部分是每个用户卸载到每个ES上的概率
            a = ddpg.choose_action(s)  # a = [R B O]
            # 对得到的动作进行处理（增加随机性）
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # 向前传播
            s_, r = env.ddpg_step_forward(a, r_dim, b_dim)
            ddpg.store_transition(s, a, r / 10, s_)
            # learn
            if ddpg.pointer == ddpg.memory_capacity:  # 如果存储的经验组达到并超过memory_capacity即10000条 就开始学习
                print("start learning")
            if ddpg.pointer > ddpg.memory_capacity:
                ddpg.learn()
                if CHANGE:
                    r_var *= .99999
                    b_var *= .99999
            # replace the state
            s = s_
            # 累加这一回合每一步得到的奖励
            episode_reward[episode] += r
            # 如果此回合达到最大步数：
            if j == MAX_EP_STEPS - 1:
                var_reward.append(episode_reward[episode])
                r_v.append(r_var)
                b_v.append(b_var)
                print('Episode:%3d' % episode, ' Reward: %5d' % episode_reward[episode], '###  r_var: %.2f ' % r_var,'b_var: %.2f ' % b_var, )
                string = 'Episode:%3d' % episode + ' Reward: %5d' % episode_reward[episode] + ' ###  r_var: %.2f ' % r_var + 'b_var: %.2f ' % b_var
                epoch_inf.append(string)
                # 如果到了第5回合后(var_counter为4后)并且后面四个回合的奖励的平均值大于等于最大奖励时：
                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0  # 回合置为0  从0回合重新开始开始
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1

        # end the episode
        if SCREEN_RENDER:
            env.canvas.tk.destroy()
        episode += 1

    # make directory
    dir_name = 'output/' + 'ddpg_'+str(r_dim) + 'u' + str(int(o_dim / r_dim)) + 'e' + str(limit) + 'l' + location
    if (os.path.isdir(dir_name)):
        os.rmdir(dir_name)
    os.makedirs(dir_name)

    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i+1 for i in range(episode)], episode_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards.png')

    # plot the variance
    fig_variance = plt.figure()
    plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    plt.xlabel("episode")
    plt.ylabel("variance")
    fig_variance.savefig(dir_name + '/variance.png')

    # write the record
    f = open(dir_name + '/record.txt', 'a')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    for i in range(episode):
        f.write(epoch_inf[i] + '\n')
    # mean
    print("the mean of the rewards in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(episode_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards:" + str(np.mean(episode_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # standard deviation
    print("the standard deviation of the rewards:", str(np.std(episode_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the rewards:" + str(np.std(episode_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the rewards:", str(max(episode_reward[-LEARNING_MAX_EPISODE:]) - min(episode_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the rewards:" + str(max(episode_reward[-LEARNING_MAX_EPISODE:]) - min(episode_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    f.write("current time:" + str(datetime.datetime.now()))
    f.close()

