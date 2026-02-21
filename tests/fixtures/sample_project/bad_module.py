from math import *
import os
import json


def compute(flag: bool) -> int:
    if flag:
        for i in range(2):
            if i > 0:
                while i < 2:
                    if i == 1:
                        return i
    return 0
