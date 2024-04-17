from abc import ABC, abstractmethod
from typing import List, NoReturn, Optional

import numpy as np


class LossLayer(ABC):
    @abstractmethod
    def loss(self, pred: np.ndarray[float], label: float) -> float:
        pass

    @abstractmethod
    def bottom_diff(self, pred: np.ndarray[float], label: float) -> np.ndarray[float]:
        pass


def sigmoid(x: float) -> float:
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_val: float) -> float:  # sigmod 미분
    return sigmoid_val * (1 - sigmoid_val)


def tanh_derivative(tanh_val: float) -> float:  # tanh 미분
    return 1.0 - tanh_val**2


# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a: float, b: float, *args) -> np.ndarray[float]:
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    def __init__(self, mem_cell_ct: int, x_dim: int):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct  # 150

        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)

        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)

        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))

        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr: float = 1) -> NoReturn:
        """
        :param lr: learning rate
        """
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)


class LstmState:
    def __init__(self, mem_cell_ct: int, x_dim: int):
        self.g: np.ndarray[float] = np.zeros(mem_cell_ct)  #
        self.i: np.ndarray[float] = np.zeros(mem_cell_ct)  # input gate
        self.f: np.ndarray[float] = np.zeros(mem_cell_ct)  # forget gate
        self.o: np.ndarray[float] = np.zeros(mem_cell_ct)  # output gate
        self.s: np.ndarray[float] = np.zeros(mem_cell_ct)  # cell state
        self.h: np.ndarray[float] = np.zeros(mem_cell_ct)  # output values
        self.bottom_diff_h: np.ndarray[float] = np.zeros_like(
            self.h
        )  # diffs for next timesteps
        self.bottom_diff_s: np.ndarray[float] = np.zeros_like(
            self.s
        )  # diffs for next timesteps


class LstmNode:
    def __init__(self, lstm_param: LstmParam, lstm_state: LstmState):
        # store reference to parameters and to activations
        self.state: LstmState = lstm_state
        self.param: LstmParam = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc: Optional[np.ndarray[float]] = None

    def bottom_data_is(
        self,
        x: np.ndarray[float],
        s_prev: Optional[np.ndarray[float]] = None,
        h_prev: Optional[np.ndarray[float]] = None,
    ) -> NoReturn:
        """the result of ht(bottom right)"""
        # save data for use in backprop
        self.s_prev: np.ndarray[float] = (
            np.zeros_like(self.state.s) if s_prev is None else s_prev
        )
        self.h_prev: np.ndarray[float] = (
            np.zeros_like(self.state.h) if h_prev is None else h_prev
        )

        xc = np.hstack(
            (x, self.h_prev)
        )  # concatenate x(t) and h(t-1) (first merge in bottom line)
        self.state.g = np.tanh(
            np.dot(self.param.wg, xc) + self.param.bg
        )  # input gate layer(third-one at the bottom line)
        self.state.i = sigmoid(
            np.dot(self.param.wi, xc) + self.param.bi
        )  # input gate layer(second-one at the bottom line)
        self.state.f = sigmoid(
            np.dot(self.param.wf, xc) + self.param.bf
        )  # forget gate layer(first-one at the bottom line)
        self.state.o = sigmoid(
            np.dot(self.param.wo, xc) + self.param.bo
        )  # output gate layer(forth-one at the bottom line)
        self.state.s = (
            self.state.g * self.state.i + self.s_prev * self.state.f
        )  # cell state(center at the top, Ct)
        self.state.h = (
            self.state.s * self.state.o
        )  # hidden state(last one at the bottom, ht)

        self.xc = xc

    def top_diff_is(
        self, top_diff_h: np.ndarray[float], top_diff_s: np.ndarray[float]
    ) -> NoReturn:
        """
        backpropagtation process in LSTM
        """
        # Calculate the gradients of loss in gate output
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # Calculate the gradients of loss in gate input and activation function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # Calculate the gradients of loss in weights and biases
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # Calculate the gradients of loss in input x
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save the gradients of loss in input x to the previous node
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim :]


class LstmNetwork:
    def __init__(self, lstm_param: LstmParam):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_list: List[float], loss_layer: LossLayer) -> float:
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(
            diff_h, diff_s
        )  # update first top line diff
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(
                self.lstm_node_list[idx].state.h, y_list[idx]
            )
            diff_h += self.lstm_node_list[
                idx + 1
            ].state.bottom_diff_h  # add the previous node's hidden diff
            diff_s = self.lstm_node_list[
                idx + 1
            ].state.bottom_diff_s  # set the previous node's cell diff
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1

        return loss

    def x_list_clear(self) -> NoReturn:
        self.x_list = []

    def x_list_add(self, x: np.ndarray[float]) -> NoReturn:
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)
