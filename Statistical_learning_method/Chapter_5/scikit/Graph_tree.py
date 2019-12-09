# -*- coding: utf-8 -*-
# @Time  : 2019/12/9 20:11
# @Author : Mr.Lin


import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)