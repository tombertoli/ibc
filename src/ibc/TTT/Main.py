# -*- coding: utf-8 -*-
from Historia import Historia

eventos = [ [["a"],["b"]],
            [["b"],["a"]] ]

h = Historia(eventos)
for _ in range(100):
  h.forward_propagation()
  h.backward_propagation()
print(h.posteriors())
