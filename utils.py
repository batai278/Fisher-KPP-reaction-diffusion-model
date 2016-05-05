# coding: utf-8

def cast_init_to_fun(phi):
    def initial(x):
        return phi
    return initial
