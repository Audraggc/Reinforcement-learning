__all__ = ["valor_inicial", "mostrar", "crear_politica_aleatoria_estado", "crear_politica_aleatoria_estado_accion",
           "ejecutar_juego", "implementar_politica", "evaluar_politica", "mejorar_politica",
           "iteracion_valor", "iteracion_politica", "caracterizar_entorno", "plot_policy",
           "plot_state_value_grid", "plot_action_value_grid"]

import random
from time import sleep
import numpy as np
from IPython.core.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns


def valor_inicial(env):
    """
    Establece el valor inicial para la función de valor dependiendo del tipo de estado

    :param env: Entorno de OpenAI Gym - FrozenLake
    :return: Valor inicial en términos de la recompensa
    """
    terminal_state = dict()
    map = np.asarray(env.desc, dtype=str).flatten()
    for i, s in enumerate("".join(map)):
        if s == 'H':
            value = -1  # Death state
        elif s == 'G':
            value = 1  # Goal
        else:
            value = 0.1
        terminal_state[i] = value
    return terminal_state


def obtener_Q_inicial(env):
    val_init = valor_inicial(env)
    return {(state, action): val_init[state]
          for state in range(env.observation_space.n)
          for action in range(env.action_space.n)}


def mostrar(env, wait=0, mode="rgb_array"):
    """
    Muestra el entorno a modo de animación

    :param env: Entorno de OpenAI Gym - FrozenLake
    :param wait: Tiempo de espera en segundo para realizar el siguiente paso
    :return: None
    """
    screen = env.render(mode=mode)
    plt.imshow(screen)
    plt.axis(False)

    sleep(wait)
    display(plt.gcf())
    clear_output(True)


def crear_politica_aleatoria_estado(env):
    """
    Genera una política aleatoria en forma de diccionario por estado
    :param env: Entorno de OpenAI Gym - FrozenLake
    :return: Diccionario correspondiente a la política
    """
    return {
        state: env.action_space.sample()
        for state in range(env.observation_space.n)}


def crear_politica_aleatoria_estado_accion(env):
    """
    Genera una política aleatoria en forma de diccionario por estado-acción
    :param env: Entorno de OpenAI Gym - FrozenLake
    :return: Diccionario correspondiente a la política
    """
    policy = {}
    for key in range(0, env.observation_space.n):
        p = {action: 1 / env.action_space.n
             for action in range(0, env.action_space.n)}
        policy[key] = p
    return policy


def ejecutar_juego(env, policy, num_iterations=100, wait=0, mode="rgb_array"):
    """
    Corre la simulación de un episodio siguiendo una política dada.

    :param env: Entorno de OpenAI Gym - FrozenLake
    :param policy: Política para seleccionar la acción, esta puede ser una función, un diccionario de estados
        o uno de estado-acción
    :param num_iterations: El número de pasos a correr.
    :param wait: Tiempo de espera en segundo para presentar la visualización
    :return: None
    """
    state = env.reset()
    for i in range(num_iterations):

        if hasattr(policy, "__call__"):
            action = policy(state)
        else:
            current_policy = policy[state]
            if isinstance(current_policy, dict):
                probabilities = np.array([current_policy[action] for action in range(env.action_space.n)])
                action = random.choices(range(env.action_space.n), probabilities / np.sum(probabilities))[0]
            else:
                action = policy[state]
        state, _, done, _ = env.step(action)
        mostrar(env, wait, mode)
        if done:
            break


def implementar_politica(env, policy, trials=100):
    """
    Get the average rate of successful episodes over given number of trials
    : param policy: function, a deterministic policy function
    : param trials: int, number of trials
    : return: float, average success rate
    """
    env.reset()
    success = 0

    for _ in range(trials):
        done = False
        state = env.reset()
        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            success += reward

    avg_success_rate = success / trials
    return avg_success_rate


def evaluar_politica(policy, value, trans_prob, reward, gamma, max_itr=20):
    """
    Policy evaluation
    : param policy: ndarray, given policy
    : param value: ndarray, given value function
    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : param max_itr: int, maximum number of iteration
    : return: updated value function
    """
    counter = 0
    num_state = policy.shape[0]

    while counter < max_itr:
        counter += 1
        for s in range(num_state):
            val = 0
            for s_new in range(num_state):
                val += trans_prob[s][policy[s]][s_new] * (
                        reward[s][policy[s]][s_new] + gamma * value[s_new]
                )
            value[s] = val
    return value


def mejorar_politica(policy, value, trans_prob, reward, gamma):
    """
    Policy improvement
    : param policy: ndarray, given policy
    : param value: ndarray, given value function
    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : return:
        policy: updated policy
        policy_stable, bool, True if no change in policy
    """
    policy_stable = True
    num_state = trans_prob.shape[0]
    num_action = trans_prob.shape[1]

    for s in range(num_state):
        old_action = policy[s]
        val = value[s]
        for a in range(num_action):
            tmp = 0
            for s_new in range(num_state):
                tmp += trans_prob[s][a][s_new] * (
                        reward[s][a][s_new] + gamma * value[s_new]
                )
            if tmp > val:
                policy[s] = a
                val = tmp
        if policy[s] != old_action:
            policy_stable = False
    return policy, policy_stable


def iteracion_politica(env, trans_prob, reward,
                       gamma=0.99, max_itr=30, stop_if_stable=False):
    """
    Policy iteration
    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : param max_itr: int, maximum number of iteration
    : param stop_if_stable: bool, stop the training if reach stable state
    : return:
        policy: updated policy
        success_rate: list, success rate for each iteration
    """
    success_rate = []
    num_state = trans_prob.shape[0]

    # init policy and value function
    policy = np.zeros(num_state, dtype=int)
    value = np.zeros(num_state)

    counter = 0
    while counter < max_itr:
        counter += 1
        value = evaluar_politica(policy, value, trans_prob, reward, gamma)
        policy, stable = mejorar_politica(policy, value, trans_prob, reward, gamma)

        # test the policy for each iteration
        success_rate.append(implementar_politica(env, policy))

        if stable and stop_if_stable:
            print("policy is stable at {} iteration".format(counter))
    return policy, success_rate


def caracterizar_entorno(env, samples=1e5):
    """
    Get the transition probabilities and reward function
    : param env: object, gym environment
    : param samples: int, random samples
    : return:
        trans_prob: ndarray, transition probabilities p(s'|a, s)
        reward: ndarray, reward function r(s, a, s')
    """
    # get size of state and action space
    num_state = env.observation_space.n
    num_action = env.action_space.n

    trans_prob = np.zeros((num_state, num_action, num_state))
    reward = np.zeros((num_state, num_action, num_state))
    counter_map = np.zeros((num_state, num_action, num_state))

    counter = 0
    while counter < samples:
        state = env.reset()
        done = False

        while not done:
            random_action = env.action_space.sample()
            new_state, r, done, _ = env.step(random_action)
            trans_prob[state][random_action][new_state] += 1
            reward[state][random_action][new_state] += r

            state = new_state
            counter += 1

    # normalization
    for i in range(trans_prob.shape[0]):
        for j in range(trans_prob.shape[1]):
            norm_coeff = np.sum(trans_prob[i, j, :])
            if norm_coeff:
                trans_prob[i, j, :] /= norm_coeff

    counter_map[counter_map == 0] = 1  # avoid invalid division
    reward /= counter_map

    return trans_prob, reward


def plot_state_value_grid(state_values, dimensions):
    """ Plots the State_Value_Grid """
    num_rows, num_cols = dimensions

    state_values_grid = np.array([value for state, value in state_values.items()]
                                 ).reshape((num_rows, num_cols))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(state_values_grid, cmap='coolwarm',
                annot=True, fmt=".1f", annot_kws={'size': 16}, square=True)
    ax.axis(False)
    plt.show()


def plot_policy(policy, dimensions):
    font_size = 18
    quiver_scale = 15
    colormap = plt.cm.coolwarm
    num_rows, num_cols = dimensions
    observable_space = num_cols * num_rows
    zeros = np.zeros(dimensions)

    izq = np.zeros(observable_space)
    der = np.zeros(observable_space)
    arr = np.zeros(observable_space)
    aba = np.zeros(observable_space)

    for state, policy_state in policy.items():
        if isinstance(policy_state, dict):
            for action, prob in policy_state.items():
                if action == 0:
                    izq[state] = -prob
                elif action == 1:
                    aba[state] = -prob
                elif action == 2:
                    der[state] = prob
                elif action == 3:
                    arr[state] = prob
        else:
            if policy_state == 0:
                izq[state] = -1
            elif policy_state == 1:
                aba[state] = -1
            elif policy_state == 2:
                der[state] = 1
            elif policy_state == 3:
                arr[state] = 1

    izquierda = izq.reshape(dimensions)
    derecha = der.reshape(dimensions)
    arriba = arr.reshape(dimensions)
    abajo = aba.reshape(dimensions)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.matshow(np.arange(observable_space).reshape(dimensions), cmap=colormap)

    # Asigmanos las flechas que indicarán las posibles acciones
    ax.quiver(izquierda.T, zeros, scale=quiver_scale)
    ax.quiver(derecha.T, zeros, scale=quiver_scale)
    ax.quiver(zeros, arriba.T, scale=quiver_scale)
    ax.quiver(zeros, abajo.T, scale=quiver_scale)

    # Establecemos parámetros de visualización adicionales
    ax.set_title("Política", fontsize=font_size)
    ax.axis(False)
    plt.show()


def quatro_matrix(left, bottom, right, top, ax=None, triplotkw=None, tripcolorkw=None):
    triplotkw = {} if triplotkw is None else triplotkw
    tripcolorkw = {} if tripcolorkw is None else tripcolorkw
    if not ax: ax = plt.gca()
    n = left.shape[0]
    m = left.shape[1]

    a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))

    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4:(k + 1) * 4, :] = tr + k * 5

    C = np.c_[left.flatten(), bottom.flatten(),
              right.flatten(), top.flatten()].flatten()

    triplot = ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor


def plot_action_value_grid(action_value_pairs, dimensions, fig_size=None):
    if fig_size is None:
        fig_size = (8, 8)

    """ Plots the State_Value_Grid """
    num_rows, num_cols = dimensions

    state_values_grid = np.zeros((*dimensions, 4))
    for pair, value in action_value_pairs.items():
        state, action = pair
        row, col = np.unravel_index(state, dimensions)
        state_values_grid[row][col][action] = value

    left = state_values_grid[:, :, 0]
    bottom = state_values_grid[:, :, 1]
    right = state_values_grid[:, :, 2]
    top = state_values_grid[:, :, 3]

    top_value_pos = [(x + .38, y + .25)
                     for y in range(num_rows) for x in range(num_cols)]
    right_value_pos = [(x + .65, y + .5)
                       for y in range(num_rows) for x in range(num_cols)]
    bottom_value_pos = [(x + .38, y + .8)
                        for y in range(num_rows) for x in range(num_cols)]
    left_value_pos = [(x + .05, y + .5)
                      for y in range(num_rows) for x in range(num_cols)]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_ylim(num_rows, 0)
    tripcolor = quatro_matrix(left, top, right, bottom,
                              ax=ax,
                              triplotkw={"color": "k", "lw": 1},
                              tripcolorkw={"cmap": "coolwarm"})
    ax.margins(0)
    ax.set_aspect("equal")
    # fig.colorbar(tripcolor)

    for i, (xi, yi) in enumerate(top_value_pos):
        plt.text(xi, yi, round(top.flatten()[i], 2), size=11, color="k")
    for i, (xi, yi) in enumerate(right_value_pos):
        plt.text(xi, yi, round(right.flatten()[i], 2), size=11, color="k")
    for i, (xi, yi) in enumerate(left_value_pos):
        plt.text(xi, yi, round(left.flatten()[i], 2), size=11, color="k")
    for i, (xi, yi) in enumerate(bottom_value_pos):
        plt.text(xi, yi, round(bottom.flatten()[i], 2), size=11, color="k")
    ax.axis(False)

    plt.show()


def iteracion_valor(env, trans_prob, reward, gamma=0.99, max_itr=100):
    """
    Value iteration
    : param trans_prob: ndarray, transition probabilities p(s'|a, s)
    : param reward: ndarray, reward function r(s, a, s')
    : param gamma: float, discount factor
    : param max_itr: int, maximum number of iteration
    : return:
        policy: updated policy
        success_rate: list, success rate for each iteration
    """
    success_rate = []
    num_state = trans_prob.shape[0]
    num_action = trans_prob.shape[1]

    # init policy and value function
    policy = np.zeros(num_state, dtype=int)
    value = np.zeros(num_state)

    counter = 0
    while counter < max_itr:
        counter += 1

        # value update
        for s in range(num_state):
            val = 0
            for a in range(num_action):
                tmp = 0
                for s_new in range(num_state):
                    tmp += trans_prob[s][a][s_new] * (
                            reward[s][a][s_new] + gamma * value[s_new]
                    )
                val = max(val, tmp)
            value[s] = val

        # policy recovery
        for s in range(num_state):
            val = 0
            for a in range(num_action):
                tmp = 0
                for s_new in range(num_state):
                    tmp += trans_prob[s][a][s_new] * (
                            reward[s][a][s_new] + gamma * value[s_new]
                    )
                if tmp > val:
                    policy[s] = a
                    val = tmp

        # test the policy for each iteration
        success_rate.append(implementar_politica(env, policy))
    return policy, success_rate
