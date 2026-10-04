"""赛博果蝇悬浮窗(对齐 BV1fQYB6oECS 实机设计)。

    & $py tools/fly_overlay.py                 # 启动(配置存 .cache/fly_overlay_config.json)
    & $py tools/fly_overlay.py --state .cache/ann2_state.json   # 挂真实训练活动度
    & $py tools/fly_overlay.py --reset         # 重置配置

独立程序:dist/FlyOverlay.exe(PyInstaller --onefile --noconsole 打包,双击即用)。
    - 脑布局数据内嵌于本文件(gzip+base64),exe 不依赖仓库;
    - exe 的配置存 %APPDATA%\\FlyOverlay\\fly_overlay_config.json;
    - 崩溃日志:同目录 overlay_crash.log / overlay_render.log。

快速上手:双击启动 → 右上角出现悬浮窗(点击穿透+不抢焦点+游戏捕获看不到)
→ Ctrl+Alt+D 进入调整模式(拖动移动/滚轮缩放)→ 再按 Ctrl+Alt+D 锁定。
全局热键(穿透时鼠标点不到,全部走系统热键):
    Ctrl+Alt+D 调整 | 方向键 移动 | +/- 缩放 | C 穿透开关 | X 捕获排除开关 |
    B/T/K/J 开关 脑图/统计/波形/控制台 | H 隐藏 | R 重载配置 | Q 退出

布局对照视频(2026-10-05 逐帧核对):
    标题栏 FLY | 神经活动 → 状态行"● 神经活动报告中" → **左脑右蝇**(琥珀色
    脑点云 + 果蝇插画与指示箭头) → **键帽行**(W A S D SHIFT SPACE,按下点亮)
    → 神经元/突触统计 → 3×2 分组数据格(带进度条) → 控制增量(鼠标 Δ) →
    累计统计 → 底部全宽波形条 → **JSON 事件控制台**(真实输入事件流)。
    与视频一致:脑云静置微旋、琥珀主色;与我此前版本不同:无大字动作行,
    动作信息由键帽 + 事件流承载。

交互(全部真数据):
    - 键帽/事件流 = GetAsyncKeyState 真实轮询;脑云分群随真实输入激发
      (移动→VNC 运动群,鼠标移动→视叶,攻击→cb_motor),页脚诚实标注。
    - WDA_EXCLUDEFROMCAPTURE:屏幕捕获(果蝇的眼睛)看不到本窗。
    - 几何/开关持久化;调整模式中途异常不会把穿透状态毒化进配置。
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.wintypes as wt
import gzip
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IS_FROZEN = bool(getattr(sys, "frozen", False))
# 独立 exe:配置放 %APPDATA%\FlyOverlay;脚本模式:仓库 .cache
CONFIG_PATH = ((Path(os.environ["APPDATA"]) / "FlyOverlay")
               if IS_FROZEN else (ROOT / ".cache")) / "fly_overlay_config.json"

# ---------------------------------------------------------------- 内嵌脑布局
# (gzip+base64 的解剖分组布局,4,178 采样神经元/27 群;exe 模式的数据源)
_EMBEDDED_LAYOUT_B64 = (
    "H4sIALWawmoC/6W9y64kSXKm+S65TjRM76rcz7Y3s2wQBLtYJGrAyWpUFYlpDObdx8K+T1z13CIii6hE"
    "nfBz3M3NVEXl8ssvIv/vL//5yz+kX3/5t7/8+T/+119/+Yf/8cv/8d//z19+/eWf//qHP/72L3/67d/+"
    "6bc//sdf/vzb/as//M9/+uO//usf//LH3/7mq9/+5c9/+MuffvsjL//029/uF3/90x94+X//+W9//gv/"
    "/Osff/vrn//yv9+8+Ke//c9vb/yXP378og+/873x7f/0urnzl/tT92///O9vbud+ue8hbuC8yoff+ZXx"
    "+zcX/88//fU//vnf/+n+xf0N//of//bP/75/+b/+8uf/649/+Nufngf58Duv+p+//eFcyuflsZbfXp93"
    "/+11rOa3f+9HOV4dl/72r3/89Zf/597N67+Vkcuv139rueb7R51X+vajzudHG/X+UeZs94+ey7cf9SrP"
    "B1L/9rc6xrdXea7xvLr6tx99fnvVuFhb69tb6nouVlOb33452uID397SU/v2ljL6t29IJT8f75df9O3O"
    "cs7Pt/cynw/U53PluYmeyvPtz/21/Lz//nR/3vE8SZvPO9rgIrUWvjVzK30/Xs+NH6k+V6nfXpXOely5"
    "P6+e99f13NfMLBULkFt63vF8rDVWszyPXJMXqf15rGcZyvXtY4lbuBf6WanxfPd9l89i1vW8f8znVWKf"
    "WOd7n7jUt7XsVyx3fh7uuVZr43nwOp5Hrfn5dL+ez9379Dzq+Pbxxn318rylleZdPh93uXPjnZO1YfnW"
    "87f7UZ8tnMWvfe7hearnq/vFj5TcAa6k8LmUxVXmEhc3ckpWa+2RyPU8XEnPfrBjrVZ3ej3Xer47jfV8"
    "z3q27BZyZOj5XX+Ezc25ReiR4/UIQWFT76dA4q+677KuOtgWfvmsRWkLqa5I9bN4tT3LVXjWmp5j0Cry"
    "n9ncUp4l8mvb85B5Paeh9cVNXN8+nqZfN5+nq27Lc+XSeeTnx73Cz6cfwWorPc/67HAviOP1HMT+HIVe"
    "nzeU2IjKAUQOQ2iex4/nbo0TOzhCjzjei/4sTeWa6dEXObGBlaO3ipL+7WL30yMtgyVFC1yFXUIZzOeL"
    "Wn/u1mOW8vNFuVzcYEOcK4vx/Jisb3o+nVnDstgsnrwhio2db41HT4/cJ+SmluRaomYqX3Mot1tDcpHi"
    "Ajzilheay8fqzxOk4pXLs7bDj6M1MiqLN47Jx55Pc0PsQRnPmtTM41zPjzpReL2jbQf6jht5ZKE21gQl"
    "cLmDz+7ci4BUInKJjZ98Z+rHVt+vHhHhxy2NqqpH5jJKnSW8nmvdH362H7FpLGjzLQhKas9ZDSm4nt3J"
    "F2v+vLGice7j3lHDz/o8j50HVomlvh4VeJ8t3oG2S2wYX1YSUl9Y4+euMvd4b/rzaOVRhLfmfES0jkcR"
    "XtklmceNoPruHWX7Ut9nqK7CkWCBrsdQVfdjPCfqVtRI10QrXQ319Vy6oEDTQCM8P7K6qnET12M00kjj"
    "ORLP5+57YYEfvdcK+ggNdL/zWQeuhdjfVoVz9Wxueun5jllxQ549KGxuf1R2eWxmba7p83TZX86C9ntW"
    "8zauzxPwt1v1+ONZuIsd9McjibdNQouhqtvzHLd8PV/ghvZHvd7GlaM6VDNojWcVbwlAsp6vSx7Zwkqt"
    "oTwfnkFFt69H4jPbm+ZEvbAMCwdkPUt6GzC0WsI/6JzLhfVYW2ncu/vc7PUo5zzQwIuH7Bw+Viqxg5WP"
    "h8mtDZeDHVzPPWfWJnUOEX9qU/vwSC0Ccx9WLpk55M/5z2xWajzA87vbDLGf6GM3cqEGruZ5e557cv1H"
    "9DiW2PiBieGc3Mbx2fWMDii4Nw0ThnXJnLb7ux4RxWup/TEkt0bA8uC8tbBK2FF8seLXLoy+71zcsb98"
    "bi/NSz/ludmB9XPx0qO37z3Cx0rcWeI+sVFdzTjRDXX7dyVVlnKiDdA6DfFHtVRU9S2HLBgHn4uUR8Fn"
    "3YOFY5f0mbFz+fmCe/+eHzxknY9+un0glPpzs7eHg5jpLKI1S9lXLriM94nFWHIy9KP12xcuFx5n8/3P"
    "nSTVCvoS/XWLKEYG7427u/UlN7K0uOxEUdg8o8+do+Fvf5Nlfh6gIJY8zEL28RVqrOcjLvXZjuq+GbWM"
    "i9XFLcc/Y29i4fOjHdqs2OCBohqouYRR5VbLowIyAYY69tb+z8W0BePRbDWHOnkOMf7NvUJ4SZwTnIr7"
    "xjLrhRXjrKFC7zXXeWU32XdvEK8S45oqUREqrWXXdRIUjX2kk45T1flG93F44gPsbcYaJ1TyI2q5P95a"
    "xVW4VRIGGjutA8ylKic9s7J5YJpKx73z/wuffn7gs1YehrNS+3PGUn62Nusv8g7Uc9JzXdgItMR9FjlU"
    "3GtG38bBfr7mVkA4Jhzi/tzkrQKem5zLSGFtTV5X87Cgn/Wb8DZxHfDq783kuRuihAonvlTtVb1B9GeE"
    "YBVrd+FU6UFMnDbCwFsFGEtykHDh8NbGxFayE0TP6LD7DQXNv4i2WUWslSKBh5wLQpC1xXgvOXPlR4xz"
    "xUg0dwmh9E7w9XpHU6MrG2ehDaIuVMHkJBZWjzNHzI09Ut2ErXik9/YJdLvaNmfhDKIcKrquDdRhHvrc"
    "egNr+z/353B5jlCoEDR0zGDDocu4TQVfIi9jJwxHxaQSBulglbwQ1uehFlq0hV/CieZana1ahASFs5Bw"
    "a8BCiAHuYAZvlfO9UCe94BqwHTNih0HY0w/3RE8WueA8ofhvF4/7QcYuZHq63RyvSzcx7+jv1pVgOUO9"
    "rMJlS1kEYpPu1ia2ZSIRCYfPr7sQq4F84LklnDk81UfQCJGKu4aoJOz2/c3+jWOHXm7PB+6wsz8RzWMY"
    "Cy5KCT936bZm9JqYEs7vagdag7nJ7bnjhDev1koc5Vt20LrIvmHeYukbUWVqY3s3raiblObMArNeOXxu"
    "P8DdNqw5gdTChQR2AoS6lWUi1kKhIydcpPkKpCljtG+BWjusNN7PKGDl/ladxugo/8ptIkXaKfx4Tfgd"
    "iBCkEXmLZRmjD9zM8EQvzCPaCegsoyczEUni272zxDHNEVMhWmx+4nQa6rGhhCo4oqifOyoBJOCh6vOp"
    "5APn5xFviVkAQgRG3Xj6MZWXrlQnKOSrC35vU8RQvJUHvpC0zooaxnlGCeE7fhtBHy68iATnu4nnEGk1"
    "rfP00xwr1phtEAbKvnNwxxNlmfQPtT54WZgF7XpLAaqg7PPxfoz2bcIACbLy27cnnbA0d4DPIiQ2saGN"
    "EFmwyxRq+Pm6e59ZEz5wTaQaNdPFKdEMiF7GV8+gC7fPx1physMmo4FQGJzKVjfAfIuJgR0KBonA4s8L"
    "u4l96vg/uruGKn4l3pyok3DTfS2icFQyTks7vJQsgJNYkYoJ5rIF/ff4KLeJJsbFDx6PuDZAgdyM//Wx"
    "vOFFsEnAIN45PBV4S21HFg29VglPalJFiNRPQSbMLFHwQID0vHixcE0yPzgNGczsFj+0usqYXxp8EHV0"
    "3LE7EPRBxpaARDCcdb+JLjlQaMacDW24BIFYxk/rCCt6tXOoUdIJ1+fWemCZupgENVUnYyFEl2rIWGUd"
    "4emFedIZw/fLa+5MRjFia+EPoh+z3s4Budxyj89Q1M4TnVUOtKAKnXbjFCARkIoKiF06fsVUjwNxIeXT"
    "lawbAdaDvlU14IcpBhXm8pcC7/UAiASplmdruraIwdLSpsO7XsKwS78Hr0kpMnZfWBa2EmT7fufzYxAL"
    "KCGYBIWPMC8XYiuNKdhWLaATBB5Vn6gvEYO5JVKMQw8xgQC+8ja6Tx0lIDyHWLP1FQccCDkRGdxr++wo"
    "Xt19ShF5kGFsY8ZUNJ1nUw+X3jkw2EDJotN7VQUmI4P1is0G98oVM3jv7UXMDX57WhXswgG7txU9UPVo"
    "2F5MB+LXMpYYq3RbKlRF3dmr29khhMC55U84yMIm4qwNXdIAGHLF4RJuGMQ+jzK45TJtpyivcjj19xHG"
    "eeBviVvWO3fVqi4luo7cQSMWBI6rwLrGhzmJIYrsA6WGZsUDNHOITBSeuODy+q1KT/bpMHp1cn84uQ0h"
    "v7+Pm+DS3MPgaKFMCxBfYTGv9eYM41mwCuHJ6nqx4WDctxrkO9nGtFA7evxoCHR1E4pALQOSJDyqezE4"
    "iEbtZMKeP93eM2F4y9vDSBi9S9XAMt+HlL8Ro6xHdxfUZwKATORMshmqohvBl+dsqgWxD4RGY4Uji0ZS"
    "Cbt6Qv46wATSmlRwwzyFLTz3SzctHd6aSDcQgsHFrYqxotU8FFLH/2PO+plWaq4puJPGEOG+PUT07FKq"
    "2X5E6cq6y6gezPNqpmrmxsRv3Ta3/jPo0acixUpKpOVI8wEsApVd9fCz7oi4vJI/2vs8vK6OPskMTj3o"
    "jlhSXuJFgrumsNCBOJ/3YxBnAT1mXfqiW4I/krZCLFinmrS9BwiWtBz4N8TIrtUA7zKmViwmNzdQkgUH"
    "zmyCEcxOOZeiI483WTBd4cYNU3S8E68L96lXY1POQzfQQnyQqUTEGbgHtutWazq2cy+BIQ5x9uREAmrc"
    "JgjQHaPYZ94WuZMuq6beUJH3hnKEcUb6pWtLXBaZOHxaMh2J9eE5yKoqB2dIQRhbSAjcJ4EPoebEiMzX"
    "9LVTLK24IGMbi1tSzaZgLPBfSJy80o9dlJzFJdkJz6IOV1yIjBta2prnJFUhzBZsELcPkWH9GzpngrWo"
    "vtswZ8QSqoZJrqBPONMXGg6zYih4RwrYUTCQ1dshkiNwU68P/MtGk/vJpGazCrJpsUCO5IuwpTNAc6JZ"
    "1JDRoyBKZbkFFHBQhFarERsQN34UNJt2RQQjpoo2wvdRp5nXL+ELH5yFJhLWZN1gc9BHZq2aNt30LT5g"
    "0QnAq9d1LKEn8ga8s24F8DHXCHSeKODWQOnAn4gMEyIbu+RBMiDnXNxuCw6eX9BRCLhd0+zSDrFuwUib"
    "nlLieKNOzMxFUgqXcY1tLPO1Doglk8/PPcK2rdWK7g243n0/IkdpH9dyGQ8uQvvRt++WTFQZ9B9xYAFi"
    "SpAgqs+rjcS7ywCMkYlHGisgZ8JO3yevbkaQarSZaRvQntJzl92ckcarkYERZfCkLxUhAlLV+Fn8EZD+"
    "eUNf2zlQflVhDX/+9oYFBvPGAhqSkQHCEsikAlvwYFlsoqmS0c44yujQ3NnE1jbFqoF9v3wjohkYDv0y"
    "my2WpJpL+E28k/OQIzjEKiYx2JD2sdNnEj2SpDdoCqorUxsN2Njswv0nxSVtXPT+k7AeYsbpunAccWgU"
    "LBOuYbbN35imx49sOoRq1wsDpUMIKyCBr5hAvXWauW7OkjQg9JJ4Kjy2XF1keQq4AgcjzuNVikYplZ2k"
    "uP2VtGkAQKVF5A2cK8PfKkIlYr9rGhwQwIG4AV6pjZsApulNcHlPhscdRWo2A6knWa3nIz1F5ZoMfDh6"
    "hUfM8qO8ZRTi7UbhdPA3nOLsB8iLSnVRU3nyIBk0HS38ZHZRRKKzQLdbT1AjYKCUdqKTbC5ExGeHXncQ"
    "kbeybxXdulwujIP5Sbxsc7coIPOzKE61ViFHnYeuDba+L49tOeKSS5cFmw1oLPyhcehYpJo1pgSuONmo"
    "jMh3yEmZZkTq5rLciomnwYUhJQJpK3ZqGTKbmoqIvh4JpEv/uhtBpp1XuTUb4LyZOTDOiK+4dQLQTBCU"
    "gbpLMavG6QdCKWTcMrzOe1sQs6Hj5FYAdaMgIv2YWeF20GtuncgKq3ONkOAVmb1Uu3m7PGYDyboDfzxd"
    "75qs2RUPfR4Owi0zRSo5RaAUT4U20aV7dPe1xNUJHWQLV3N9AprPpUsEdBnLDQaoJQUyJ2kMaSFBs7pl"
    "61B/AbBk8gAABoW4IWWC3QWSieBl4JN+qcKJGEVfCRgk2GRcyErMl1T95v6X3lyuO+lc4Wfi4lUzCTn4"
    "Oae24lRdMjB1INdOteUp6WrkI63M8+cg9iTDTYIiyVNI2DigJKG+HMeba03ThyREWGA5uunymLPxwyha"
    "TQw4T1a6JNHqcSANOfJ0dRu+i5OdyEJkKKGZPGgCkBGzxEEwdYOxvqUFlTSM2ER558a4bukcO4wTPYhU"
    "FK7d7ToCIhJiQEwiVG6CryUyeJBpoJpmsQENF6imUB85pZDLJhETd9FUGk6s/rqW68KyGH8FBQn3b5jX"
    "DPdvQwqJPKgHSjlsMMNueZFnJUsAnG0ecZdLVJJ5v/DpuHTSL107z5jVp0ZJQ34suKGpGiKvDC+1yYLV"
    "dSXSr0nXmCMOSguGxMqoSCC3y6ZFuQuGYCAj4Qv+YcImD533fnjQGY5Pgwh8+8MY1uugl2X9WxU5zNCM"
    "XU4g3feJxnRhACA03dodRaetE5AX91V3sEDQJVJ2+6VbsYNqTbkZ+KyWFnDr+OTJDIbGmjAdgDNNL2ic"
    "LTCj0GBukzrJtAncGlYDHKQINdZNGzELessIm63PQBxOGKDrrBoV3cmQJC5Yjg26cl+PShs4dpNszgRP"
    "0U+d8KUGKnaQFxicmUkYNQF/B7Ix57PHgyMxkbg7LjW3/1wMK3Y7VNBQnjvr1gr4N+DPoQv1XHLikAwS"
    "CQNK2cSkNZj/A/LzmLzCjx546hNUaoArz3gnv/RgkD+enM7hAca5w62Z61EoAwbpwPwP4LQLbd5Z8A66"
    "d7sZBvvfLrmu5zTPXlgVvo5QcHJ+Bnp/ADAPUmvgSwNvdsGMnlAcJ0UisUH8PwpyyIWcjyIeWYLvc5EJ"
    "EWWg2idOSsMwThCDiVM0CRonmm7CUZ9wNqYcezzATsZ8EOEOTHyTtoWvOAjHJ3ThiQ/XLQiambdcrF+s"
    "w/PtbCjO1wQNnxwqPbJJ2nrCVuik4iahd+umdR+tMCEfTExYdz1YgUGkMIm5OunRsR4VMwFNxhI3LD4Q"
    "ovZojAGKPlEPlb0f3TuD6IfnOPAqQ6aRoCFix/qj0W6vAyGGrushQwUOa3vAMRpMtwGm1xHYW/4az8WT"
    "eMbY0Xmc6Q7ncPLpOfwBWQBBHVjhsUzfWnqzFNTMWeNYleMhK2qvY8MGyzhJGdzfWvmRWXDu9nFlBqSc"
    "gc2c+B9+fICvD/zUMWQwPFs6ibQnKfcJ/2bACqT2YZARHtSgdPKig8K8QY3IUGBxpzong9PY1W3c3oQ7"
    "M/BN5paXDvjQAV5vCX6kbuBSwuEYYEIDURzQY6ePRoJiwLSZS1VDaKpIwasY5GCrsoRpGDqo0yMda8hN"
    "PGpiwnicFLUNyD/ds0J2aGIzJ2d5QrKylEMV0D1+l2cL045l4E5wb8lNxGLgS7FQAyA/peOs4fxPIuS5"
    "lOxnpTuO7gAxbtWHQKPgFEzZ1uh/uIkkdSb5FSorJzBEt9zyQn8D80x4fpPyg1mvg+tyqwANZuEtSBYC"
    "ADFpkMBpzSWAkcRnB2aoHpWU4FujGMWioBIaF7MyoCH0pL4Z6A3WB5guBF5l4ukphzpMqM+KgqgIT8O4"
    "si7c1jT30lhichmTgHYAdXZEcOAKTkLfAVxDvnXGWcNQL8wIaTQNwECHDO8fgL8jiBM+zygqMC6GqzKQ"
    "6gHGG4ZKrwTe9pDloPBlZZxFHIR3mlVc80GFxsDDG2Q4OqjbwCvz9E0KVO6TnLc5kTWmxPaXzj+OSh0a"
    "v0fcu4xFJSQOLzaQYHEQoU9iuQnYNIloJ6HNDNWhOlSu/SV3RknPfdjzttwDYtcQRyRlOKoyj4hylm+B"
    "VtfwcXL0hMkDen+n3u0+R3NX1U4g2FCn5Ew0R9RqzOKR1N1KHlPLzUgacie4VEtf4Ak6h+GLroRyBXp4"
    "xyl9m4wqMf5iTYn+fCoepy+K7HCRrHOal3oLr2Yq4eQrLgUCHwx4vVNwzaYNioOHdbqPg+TbBwF4HFeS"
    "bwPQYESqjHVaeGIXfhWSAjuu4z3f2qptZ7Bi1zqc0SHFAfrrxDiqDafME5x8oMrRPWOZL2BFWaEJLhkS"
    "gj8/iDQGqVfLaAex3l5ZVSVXwfphM4Scpt6qGDKY8NBXG5K82GtsiId5EqUpHAMAcQCGeZqAqife4yS4"
    "vx3rhL/D0pp2x4hTjTYNAy4EkkstXf3nUs8Xt3gmVCwnXpIU4o6vZx2wAj5JdwyKZyYuiCrMNU+4n/Gc"
    "pF5ZY1KhUytAkD91bQgZrGnvUY3Ono99b2PpbnNwwBmg7AwCk4I79dLOue+K6wmaSjZ1akPIW6hTmpk3"
    "FP4AFJ9FxodeH34IKeMJ12ASQAvWVLRyBTbsw7vUOwC8xVLf7jKGbLCEoU/LRljqwG0u+lKcdtKxAz87"
    "Etagr4PCx0rAKj1kymSTeKgaPjVnh5g3AOXvCLBtt65Tg6FJGNCpJxqhr8qRg+sG8GL/hqF8Vew0mH7v"
    "sl1wqijnXwBBnaNjftlT7KoQZE9inko10KTMYcAwn+B7A9RhJh05K2zRdpp89FboX44t8M0ATmkQdKfx"
    "u5/iDRBGOPNj2ICBAmA0DQXog+qwaa8EyuV1ieFRkDqbhDYT6sMgBp+sYAW8GlSKDQqEG6hF12sj7huk"
    "wOeyzwGJHFakowEax6awYzP77aRZic4GcMyA8TatA0y7AptqXHlrrtUA/Z/gUQNp7JHPSIeJmPpASw7D"
    "4JCijChH1QxCVifCspS9o/g7hbroLPt3GO/gkCKzQxOINrVBAG7yLVfmgTE17idMBfIoFuve5kfRwHR0"
    "trwIPKRd16TgT4BdgwiVuAXdL0424TAPj5M20c1jiEJQxbf0O9AR04NM2IV0uhK6BVa+ArGgBVTctQk1"
    "cXLJHgCsktLvIRLPFoCtzbYOHRC0b5w1LPGwP4le59yR3MTTHoQnBnQji8SVdYQzdiIhjOgE05fIAZiE"
    "mN2YJhKItqg/iaLFhDRwkokSgo2OpbFaeJBOCFcC2vi9EhgNnHB2WRAH4Hbo0ZNVmC0sH/LLOYSUepGp"
    "FVlMLgcSOgF8JwWmuvBWEJcToBgkOAbEM/UTR0/UC+ct2V8FvQzRXg3ueW0YDEKukJ0u3oZ6AtpoxCoY"
    "0wlFNhGIdMkmutHTIi/K5iK2myapcDhBA8ARRaTG0Hog3GjFGc1gMHLVUH/xwJV1Dc7D2qpjkGWdU1iM"
    "TAPNRwwYiRkGtaSXFASggVrNclE1snbRy7R7jThbVbUSa19a6SpWalkDugexqjaWATXCA+4BGnGP6tkk"
    "YwAnj4UFYaVcoa9yIHAcNx6pcmKK2Chwg1S5WwvNzY4asBMG3KoOLNClIGjMS0QDfZ+HCxOdzK9Ng6mr"
    "H/2ApIHr4EFIH8SvyHBEy1ZLjvbiWw2IJ1IuElca5o5w1jmWhh6g5jN6N5QDpDDofJbnEkHWLQZNpZin"
    "Y8km2ao6bamAMWJHbO8xcBWTcfkkRLmMZYh+sXoXj9l4iqGUVvW5MZBIAXB6fEFBmHSpYBKWjTcPHJ4Q"
    "eDn6fUc+uA0RKQHDjCW0gdsgRUA4G4a88EBfLhlOPxDg8qvt0AUk3KGgXvacCIiCXBIAOSR2A/h68OkF"
    "vMRImykL4LpLMCHTxENMNplneu7ggoglpGsYaKEYbJaW+gFby8lVYPAXZNcaxU83lHZDFG/oKa5L/1nm"
    "H+LI8hDFioJ05OC2X3n3mBiGXFjhxCPOqLLbVM2maoE5L+7gp0LxqYRFyLKSJdGVs9ylvl9zg5Fdnn9G"
    "0IqgEOfcqBwHeyTtmFbQ/D6eQyYXRC7Xbi9dTwZERJ/RoitthUba2LmovRD9rASxE9WUndmvp4y3S73B"
    "u0SNTnzrS28dpTRSEGs4OITV4vq2/CKGiYoFzgxO/vXqvKAj/RxhTGAnnh0W5Cfq1YluLZq4PGyowktN"
    "nXvbtdNGvpccAOEfxFnwtcJgltdurb8gxVKKMEmS46I8noAYS30BqRkWcrPqm0ood5HMqnBXu3KA7hU7"
    "0N2QEDqb1Ss2oVLniCHoreE6Wm85pZ7pjBC8oQKXb3wEbUHSmqjhQM04i3brE5BZFqaCp9qxpMN9WYTX"
    "XRd4GeLjb3KEUTeTvLPoud14BLlF+wc2d0Fon7brgr5Prn6O6NYGEsz28IZsVQ7+DZQW0byJl98J7QL4"
    "tBzfuj9Ub0NlR4QpEQiAGx7GVDFZUI5o4BovgKt12bSRF5dZRXwjsYzHTViXCPPjei4LNaFbirkZOjYB"
    "FBqMYU8qsX4jM7AEpslU2ARwilThbS88uQXtdkV/QUMsgvZp+zx0F2ZRKsy6ojciziWrtywEsYBfb6m3"
    "F8R1y2He/R3uszR2afr9dXMXwQwkcaXIrj7rB/BUDf2tH5mSy8bu0BVQUDGzws3CJZBfociGjEONmNRs"
    "31+Td0sLtYEYohFnVNfjUBUeQUy42U7N2GuUV9nXCoSlA6YgL5lHW7u/3ijRMEPzh1+4jV/Vqam2tKRr"
    "gz6rgVGJThxqMWwHXmgkZDj0rESXaEUUw7a1PI6AvBZ9IjMWZJwsLrdhpXCCHc+M0fHwrrKbVUx5AKj0"
    "Ro+JIZjJDY0i6grl2IZAM9J64M5V2GBsUVOo0BE4htEH08aOWsWNlE6NaDSTJNKQm0ES5DLM1DJc1Mtk"
    "M32Ed1JdjppHeR4zaVLJecFR8vk6QQ4hTC/23+CwJRkkHH8AeDz55toLiZGOS7a1whdatg7AEACVN3kd"
    "fUPrUzoMaNMd2AJEmBevZt7R2sQ1aJcphi9dCeTfBhseSSJnt3p0SRd0m6p1l7gPuzUJQ4qlWXm5TMji"
    "r6OSejrKEifns5tjJvE26UZUxVwsV8KTjfq9JaME5cDTLHQy+t8Qpoq3Ew1ZowacNIYtmaxbHmIX3CVr"
    "XiOspnC3S8iWnpM29pncsCk3epdiDdp2TPiELYp8kJXST16y7b8I8D3hha5kTQQe/mKnPH1K5ZfTS2K0"
    "RgcCIfFLys/umFmlDti+sIf13WwmQ21LvQINtIcToM0iBTWTSWPg0q7Py+KDri/pP8OeAoD6QF0mqFGw"
    "l+aZDIVdRGGbWZDVrrlTKpWjVSGhDLgL9qEcOBGT/GWjS2STCkppwrCS1JBK1Vainw4COHeCY6yT/zHp"
    "qzGqAaTug8gziWqA9lFsAlB12tGdZOnQ8xKdZPusy/0wcjskEFeywlA1zSH1x0SnfISLgwS0kkxeBfoC"
    "dIM7QxHGHIIw3j/uTBZ1dfvA46pJzOvsxuPRVgBDKn0AzaZJ6jk2xnefQAVkbo7GCHdWtgunbMjwSuIq"
    "4wT1r5Ofo28V8TjrsfG+zjYW2irfsQMAHVF26+mkadlmLLf9CuFlrW0r00AbAhqsxk48bd8h7SCzOBRo"
    "U01H/jc0Kj7IlLukmgEtG2SDp6Q2QqXOfUgFMutXKTh8EahYMC4CmXdITNFOkN3tHl1ZGRzTThxwrz7x"
    "Ogqtukq9bGqW9VUdb3LAfLXB8wA8D2g62SiRrYS21bOpVUtT0VM4N8tUXzSfVH65NCz4Ikt7HewlUxTQ"
    "bmV9mKatbE9wxXjySTPBARNJ6ykFaAxDGEED6CVdIE1C50wblFeQpFVhrSLf0YSrJZ+CcBJ20qLUlZ2s"
    "l7m5+FJDqyw17TLNlHbH0CB72MUY92/pCsmYgrQwTZuprwJcT1Ks8DrIH0ngmaZ/7JGgOGHD0wndDjpI"
    "1hKdP3hmfBFc3slbAoeXVyUbA4K3ya9R1RPTe1GtEY/geCRr8ykxh8muXrX5vgCQXWEHru+A4zeNENn7"
    "aBaDbpBTiVvfImZbByDZ6tFhLzwh0JlZZByAvBYTOJw8gXmAlR50jNJ2q0OznRWMZ/awkKf9tnupjW4G"
    "DUtdo3qkHyTeRCLcrnLkKyb0Oks/A7mAUx60RRO62Uah9pgRByNgFvq374Ve6YhSE9IcKOklMnA0n7cf"
    "l54US2TzyWaeGWCZ4BIHXDYjjPxIlGgticesIulXIG84lYS0dmwnj4fusSQKFNKSN6S/zHH0vowmPZIE"
    "I/ijpVq2YxggEUlGu7QgMy2H052OWuwrm/1JO4lQ0Wd1ioibEgckYIWjGfDcbJtCrbvNtitKrZL+bZbW"
    "WVVsQXf3b7aSGQd9vtY47233z69VQKvZ4cZMT9lM4mbR4hURdtqHppH5tmxGjMB+KQQiACfVNuN2eSbb"
    "SDRvx4PuacW+Ck/Y72+2ozt0CGSSh4IJnlHBbjOFsYsz75tLh3tjXfEwpWHtpefTAKwff8rSfW0bGzud"
    "tpNbl5hF3o26axQw4pDa33e88CgZ87XY1+mcegCiUalFbZamOB8EXliJ3vY245K1Zn6FY9aF5R+f2SZq"
    "doZFMu3nU1+0dI752E2hW7YzibMD6jwmNdibz76EwFCNmM+GgzW6J9lLn2+wJ77r48m3SxjeRLtMz1pi"
    "5pmkFtEAzD0d0sHybgNvJzyL7ht5z0b9USE/LnmpORHEe5b3k4+yldpVx9wQa0TLwGLPXOd5wMlJimTe"
    "3ZdDo3fb8IotRyfSvgu3imMrrJWlp0G0EHCZ59rZkhLdkO056dFNgkc8NsIYkx2ataCKy25fb9FYC9mx"
    "BgApL+aAr3QqBS9JILy0nRziqeYuEmHsiVkP/UbFdKMkx+rrdml6bZGHfSzRQ6HvLrWCmZVhLtGfD1g5"
    "Wrfb5dMxMTZas0tDxKam/5FgrRMoBEJenf+xrFnVRltgSj6ByK4JcNJ9igYflbRUs7/+iH4U7YAP6Uxl"
    "zye7AtbmVtZ8XBqAONAbwTBMdVGjykEoTjgR0z3wFJO7TmTojtAo42iSlOR2RXPcozV3mEyMurapxRPw"
    "/GPvVREkwC+QzNOCKDDVJQRuG8KujknAWHvyosv50mCmtbuM6b/a28EqLRg+975ZwIpiKKoJ1CSxkeBs"
    "aQcVK35ZryM096DUaKVmPqBuyk80K/QkisQBF3eetCw1diq7jlWz2e3+3I9mv/WVXM3baFWIVI4jKKGj"
    "YnDJboJZYwZRLrsUUTi6R5tGs+sXbb41raQ5cn3pnFrF/exee7SnaeSclbsaXdncVGRy2HEC6BgINzbi"
    "sqWq24LnS9a2sdoNfS3TtmLn7OBe0VWlG5TgXgZS6A2iNjErdrWNH1c9ekK31g7fyLBO7VmXqodzSHak"
    "THZLtY62GKbpPBhvTrF1sFeMstI5OgYKVSK3poWDyOKsHE6lDnKxk6dUrmrKAfexrT1epQ0L8Osxrcq6"
    "gZcxm7vKodn+l/RuDNxpkmiAbi974KRxdHkSwqPid5m0Knnz+1qx/8ByHsNRalbJ5evJtWgZb+tVhRBp"
    "siHOUMPno3VbE1KWMxW9To/RIZVG9cX11fVyZldJh6KIsU1DkgB6SbqFhsSy1nwg9yFSlilRmSQzKm7F"
    "KNuAy1FQti4En45cTQGxYGVh6OpsI2bG36ai7Rslp5uWBkvQjYhoClla9lmPTY1GnbaoaFHDxBFM+vs2"
    "Ke+vrpezt8MZLiu8trnjvJWEUPw2TL/S2PdMqNbcAF1fsV3zuP1YJGcI6Frq+dUYGOOpmrtErnVzQw4q"
    "somvdpP97rnuuKo7ei8fzYTNTnUldJlCbYelLFrpoyPKywVtZfvUxfEmzjqK/uuXGUZ6PxsV8PTzH3/9"
    "5X8zlJAaIZNBVdOl505nxV7etHD3cU3zwvoomvFuQXNyGkx5NfDpdhMmg2OjIxuRNCtWIYnKc3eoIC5e"
    "AQFOnJRiEzYbvrOVuBH0pIiI2C6e6RhMVm2KtVsNlGhQQccIW12REMrkX8wL6RplRaPZT5zmE3W35stO"
    "TXCoCNQMy2/s/+G4M/tSFxufcJVC8theEmiFEvSLdYwQghxEzrMRQwMbOfkgmkJxAWo2m50hCXhtKNCj"
    "VR87RbsgMMPC4BDb3VfgOmeRRa8l2NElOng3G8yVPSOk2m06Si14RRAWHcmqjSudEEP7EVyvaejOUjsE"
    "B86yBdulRas/2iw4wbHsATYB1dgXHO8vT+tC6GHDGSuQRU0cFzuIk0PrYCndsqTu1Ek9EjslvIS4Tvum"
    "2cuJdjM21wfRtpOSA7Ac/+UAK4TdAQWX/Ty0sSbQnK5JA3ZsZnT3mSqiNnd31IoTk+2KRq8TlFqx86Vz"
    "ZZZtbdZuYJanXU7tSXE59tF+zuU1Kc75HZlWrHa0KA7LrPahoOWEjpPT6ZwfAHu+ekRI0th6rxQnZtiZ"
    "FTtkHpknrTafefZhdfGavNVDstHc5Ww3+784OiYffTc4Z3b6cM5QjOyYbXdLSfLgbMRLcyGmvBQdC1Ic"
    "0etqnvOP7PMjhXEdw2dkbTvELVWbq9m/JLfXDhUbAxN5V5i2ujsUUzQ843zFkLO5z0QCqyrNUQDPh8l7"
    "lmWjHvLvdOKP4akxtYtmI/QaabtfjkP+CrazWPrnBClS89neLwid4fWKQWF8jLUiS1EoTagY8y4QaIub"
    "7p2mbdOFpRw7IjxY7LMOwb+QNEjSAoad7lC0V7T7nLvRbY4kqE2EWC2nVmEUDTRpWVUcUoPJs19puWz3"
    "icGwwZKxWLIpbs9HsAQ3pGS1cnEwIx13HDEzaT9DC65hN6W+0dRC4UW01xyqu4NcXAA8igOYlk3lomk9"
    "zoxLi194dGNvNrJlh5NjJvgub1HZgkPBuuDvq3SxL4VoziqyUl3jblvPY1ZvXqpzGTlX37tXES63W9KQ"
    "3bhsQ1YEfp2MYQPhF5kA9IJ2VPaAIlEqbG7n0DytOyLUAevNNsVzN7JBmmNUuy2u8ag4BDaNpBXIq6W0"
    "M6nsHqfaT/KiWbl69JKruML11U0Jy2DwaV9um8iZ0hpn66B5dIctTdtEW7psozPb3vN0QKGjboDQyWHK"
    "dDErA5TIHhSQOhIqBX+7TIEtO5ThBsQMWk7oTBv1qIHlrN0EN9svNYaAYYSI5aDnuQ0ZyL2QqC7RAI1u"
    "nRQ3Cr+IFY3YYOcMHN0IW6h5oUGbP+g7lQ2L2Dc92Uy6Gu3SI41Ax+nVl6vvYbEv4nVYPFsUF+0tva4F"
    "r4otZUuQCTlrRNCXE9jE24uN9ujM5yhOngt/w4jbUaQ2uZ3zmAAgI1iArrjDtAJz5KW4Zq2ymO0X2w4k"
    "l3LbhE9UlgOEdOpX2cPTqhOMLscDOvbZvmRtlxYVx7O2aJ31agKcCSZs3AdVtDiVAen2swY3ldEQpViI"
    "ZUt2u69jxZCjMD1j+yFOtcKlK8tGzraHZa0Q+aoCtLu4RDecpeI4ZBRrtm0jx94m+/ZFdHYYzn6JrtN1"
    "j2azG6NsM0sRCrmfLPcMkmsdcpdjdCN4WdI29aPLPh2w7btccfJzNwhURdatwmt36HU2Yax9ZLMxvEjt"
    "sJ2ycIm1Cw6U4ox5VljR4URCGgNTn1FM80Zgxwlt9l6r69XRLtMPL6MDS1fsaIFojTlXpELKBpXROM/5"
    "19NxCsMBObaaPfoPC04Z7DVDbXErvWM1KR5qv45G7I3AO9ydON4Y226bT5D84ZQY1l53kAc3GVrz0cDb"
    "THwmhwEBMVw9JBU5t99ogcAgubX0aDpXdvMBXNsSAugcX9QaHZ0LHmMKCNwe5U1HBmtnPGrObh4uScq2"
    "9W6n1wZ/sSaHje4ex8WQ0yEnU19PpE3n6zIpmg9jQDBR4LblGpO92q7BLGbHbCRbnGiNhe4i9FhQp0+b"
    "tbB1MHhqJfwoVHLW5mgsXkHbLQ7qsadR200Ykilqw9fuLEWaMjrQo2sUr7PddFnbRy5VP1DYlnUTghIF"
    "sd7hOq5Y7YPp8FBjqmHzgIghEHhgsFyPcYQRIlxHL+miy8qX4lMAHNq5E0+nLJvZIsNcQsarzawrJaHZ"
    "Ppn0pXMmiOdPloIjyOsQFLdxNWNTlmxCDraetA3bkx4qgufc41E3Eq87wRKgz4rDsHBadYpiMN04SMo2"
    "OqktBMe56v7Iu/1NbRbEWrmKjRIQ6IqfpBA5A+lQBM6tLraiTKeNSDggVVGQf0MpUqHEyv7wRY/eCWTm"
    "urCKpnJsWl3idp1SZMIXderclJy3yW4a/2ZfnrqjAzkkWKLIXNgtdhzzXTg2JWbJMu7R0ZFN4IRllk3Q"
    "w03AUGCCnFCVbQQczpj4vZljmx87J0jrhoA4m87BLzhjw9nD5lMFTUwG5ld4nJtDy9fRrrXEODwniMaU"
    "IeI1llwkjr2xpDnJf8C7TXUnBsplhXn0WJo7DHHeoXSviGeR7LQ73hY4iOJ1Zdq2GzUD8cbqp2KFTrO7"
    "sTPpqOCWNxLUrchXnGlwg2h7KwgH6NdxBCA9CSZYh1UMRmRnqOntRI02z8V48oREUPiOj8nKtfySK5z1"
    "tlkqju2xjtyxVlmIHTGR/+JAIB0vi9hmOhsydwffIkF216U6xtGLCu5tWSV7oH6SkNw6GFp2YdZ3rGd3"
    "ZNMmxVbWJPiKI1SgQxSShZmcf4HjbZcPWTFC5SKNhpCBEzi3fNqu+zrgHmfGZ6exl3Dr+nYRS2wNsjIP"
    "E46dzSN82/OciBvm9GpTLext/bc4b40oDZSnOqkH4BOHixpRLJbJDdME1LOUSzaTVBfP9dqeWZnWTZYz"
    "uxnxSt0Nsp2pkmlrYYK1XG+GLVlFCiZgsFScvKdWvKx2wv00W7sENLA+XadUO2CtiyUT5k2deI8VHVrR"
    "ns6YxPlH4jWmW22aTjahHJyIK+DoX/cYbYy0Q82aQ4tz24hjiabXazN4hXoC7mOPyooImIUt27m1S1PW"
    "00flOsm6iDvmY+wEt+k8mbl94ETborKiKg8+Uj2GtpYR7JqxWREZ+5cjU9m2E+sQeQC+AskoPChSellC"
    "CsP27OJfYy7BPMInoIbiQDX6gmTz8pejDiyHwSeIoSlYPK0hX+1orbKHFAnN1npGD1Uswo+5IaUdiiby"
    "cvC6SyoHhqZSLrbTd+pdP7SiBsljLSXEsJf8ENUprkuazruKzvzH5HrHh0vwsDK40tbFGa6lSulxHpeD"
    "OtgVo5zS0+YUBgiDGDtFpWq5Y+AMF6EFN4UeRZoEOEg4Zwaw2fkx8xj0YZLPeWhF0qCVu+QwY/iNp8GZ"
    "nDgdJWiOuRxcHU9HkAAl0WSnBGBHUDoxLRelLblBqzJkHevk4OsGyp7NxY6NARUzA4KpRHQSgBzUPgyA"
    "dAhM+aKPy9zTCOwr+RpW5jBkFs4gLuI1B9xMkTfxdFtrtIPuN848YPSAcmyKolYkSqWTg5TazusUqJcx"
    "RV1mzVDjMAzAITt4e0I4jm6BkJLxD7L03cB69G+N1Azbu0H8OjvsY6CNiFHVDsYMr0zo3tSBJ4Ezw2qY"
    "xeONcgydZmDrxSKDUIZ3PXCNGmOw8VAwGmKM4aHo2RWzYteRAw2N6nhRoVOMbiUMLPKGaDJmcY8hT4Zm"
    "4Hi7IpnAEfZhKwycTBvi7koO05UX4l3BeTp6gr0mPSHJMx1J4EI7MsMR5wwF5f2yACM4J22H2zEzAUvA"
    "yuUefOm1WxIVZ5cGaCASPcfOCUt4jsIHEyrA3wVKWiHl6HCs/Nobiyl+fY2frHoFpjAcDaJDToTqdJap"
    "m9h3+kWHxil0kTiCZp5Ja9SoNMWlY0OM0KLAR8KB0K2UcDSA8mOdhWN5p/3hzCOlYwoUR8sss8P4ZDtV"
    "O0egwvJsx/iIDCRempPqxmlQHFhnLnc54G+kXbxcgRDDBkpTKxsyTISYldYUgcU2J50chHjniMlLUNss"
    "YyDcALRaCRj+OidABLnAvDSpaLSaeVg93SQtsapLxwFvzpiy5XDvttk3eQaQuDGm4kAtczhGqewpndAC"
    "oxGxsXqhHUU5du+9BNQRD/UWaUYRRldQUVO1K2phg50/o5clmUa/rZtZFNdrIhDlIHQAsIryBbyXQ3O1"
    "jZ7XiN4sW3D+isUzwyElazOEU0yX7mmrZ4e8YYGyXiMG1UGZS0Iefp9eF/QzoZokkm36JW9rVCR74H+l"
    "XXFs5QXRpUzULE/H8B82TZXcGioBzSgGjGTggki/t2ljpLLLETsFESNG7sxXNsnZXmUFhuswsN1YqWRp"
    "a85fTWVrOVV6lq0/+iY4O64wxlU7zN5ceReg0PtBFizUuI4pd8llTfVI5kh+JvYr9LF0aqGZ5yr2VT3h"
    "ve4UUqEXkhaqhrYW8lCsDnwwd5OCTqg2Au8bPRB7UNy1bqb41MjSBQIOzLshVKTTaQrseBvuO4yUyeO0"
    "Ma5Ca648LBDLDuw2UE/buckmXGxf4gl7o89emRZniJiVNaRKfR+YPHo7jzluKpaxiGUECoG/lTRuBiqr"
    "7MIzG8MLTBdKzYorZaMaSv4qdI5ikhWvw24NZn3UW8xcKxY8WXXsp+VxqHGi2+jRirzgNTgMvXa9qWj2"
    "pBmU7Ta241iB2fKUexUjyfsr6iukdqTzGPjnPo+QNEGjz1TU5BkhiufDwiRnoiAcWAcxxzDhTjZ2UuQ5"
    "zchRjI7wTBG5jbbhzOKAP6kguDO9bU9hDZ08TLBArUPEzKdZMqkHZ7tpOFyZiufeYuw4kVASG2ybTVu6"
    "Wx5VD4eSTUEPKces1gwX0nxcAZHSkKrNHRplmjgAniWAYC6jG8bn3TDRGCDQf52RYNao3co88kw9a5br"
    "5oGZ9BGdTr2fzZOc1MEih19o421xQL5AkCnp4JnDslvPG+5DcCQNjoHX8Fh8rBzzAPMOr/ILexdpLVuT"
    "VyKq0pzraxddtE0xnBP9svMHgQUjymtMNzZQ7jvSKXYx6FGNJJe0b5be5bAwjXXbHIPcHLSWHNXuFOij"
    "UsmDUPvaXLm82jGfMQL3bLnoPMiRuYhbYkw069FXTpSpHrh2uLJCbtexQDlSQY42Q/dLRLhEOHWI21nF"
    "ZB6OmDBHf6Y3rN4pE7QcM/NMmEqLHYa2ATUeI2aL9OTqSG2tnMOhHRqscxLeOKcpHFJJYRzUaKXziDp7"
    "1pxoO/Sh66YIOBCxCKw1cyS61IqU6SPa5kjVspxUtKWfWZVuUlNcu/edYyhNyjkaICYVYwwtd5QCZS4J"
    "4NICoRJYMowUBwpeAr4G3+nV0AqueZBZ5MVYLKC3KYq7jELz0UU4oH6jGKMdAUNpR9gDy8qyzGHtr9Cp"
    "p414BIM7d7ftIBYrd67+YUnC1MMWqqEr+HpdLjEFutaZ2e/Xye94ETvKjtiNNsXybB3qQGI5e07eKkQ5"
    "4SjMaHC6HYtgeAedxFEXVz1sYfBPYHccjmDJ0kj0HJ0BK/KXjytmqnOL1cKIjXkkK19EksE6nUvv3+yv"
    "Jzac9Q/KIV+R5BFfnspIL5stK80ramxpe1uqpC+jkOE06nwQ1eTo6nFTPhSZRKsaL0lLntom1eJkIDNz"
    "tdCgs2gBLMcVZJaZFTyt7LyVtcFVcb0kU2OJt8kNmFv3Sc2JZK+Gt0bgKRK+A+0q8dB9gUJeNjvC1KTY"
    "YUzmHEZZeJz6D6YTzZp3CyVNF/W+A8sCzyAvs255biBJSm7xKqqSsp1EuzIUdTxZyyqtyaOrI/OGAysq"
    "S7V+hL3GZVZJQjhOFrLEAPTryFbnZpCZVT5zUwyTKePWTy+c9JKkSkd6hiaqc+PQkp71uAuKQXq0x+YV"
    "x0DG1DDF/GsbUbZNxCiWDMmJg6QQaWg1ljo1eKfWhVxp574L+ZBM82XDTunu0fYJh2GauzdoRexbZFbL"
    "5pEmU7gGT8GGmJs5leESFCHUGLmzzryx9b7dAL5uwo7dFIJRSNKhKEUxSRvmZaT1glByZI9MkBa/VXa8"
    "hQcys3zwdOD2xj4mutKRkTd3EE1AumRDsd/I7MlilHQ3DoaptAkDLk+0jBb1SCRUhBnGrlvNTnQW4Vdh"
    "EUBZJJD1TvBEco9CAFqJhLByFG0DNMt2fArGtF5aCAZpCsrGEFxrodf2KKoqGzuUVdlSGZoJEjEiI0JY"
    "uxH7ABPxeJeY4BAsXk6Qnhsk8bALOEkFC6y+W7AAqy4Ju4n7etzKtlahZYXbTLhVe9WoMU1dpLrzwSYw"
    "Yjj1PHsJi0tkARTX2401zdBP22O2z0BG6VAHT1nieCftJCLbTyWibAmLRDJ6BpE1K7u+oka/7WZl0dqB"
    "1u2umSFqO+NVuHUnO2f1Ieo3i0C7obUfuR6TNklnhQ7LL66URH185MthzXp5B5W1qA9R18YDZYpUGfYH"
    "1xTDXw70eO3SlPBtr350JaiefPfdO4iizbSZ7NXauhw81bmzcsLHqgNdfbtUvbJsAYIe+SEN5AyUiBvv"
    "El8O1tltNsDXpToHgjsOHrBEYWla+SC0B57pVqmlReCjOG21PSLGQuBKCBKtb3DhM8o0SXAikLfqMqR0"
    "pp1zK5cVn2JgphlEvaR6s07oI84VPc2deW+NSY6jSriUo/m+1eZ2ehArKhsajhCsBPfuRAicQr92DbNz"
    "0ANlzrYheNM4RKDyCgLOkYAsTZVjZda0B4AMdPyyduB7VJYnCxOwM7IGc43q47RLXCw4AL3MSV6uTTBm"
    "3fUcRcAV66/IRnnttGzBnjKssxSM2c9A3mJYz6HInvx9s+aj72JPyx0dhRj8jxMV1Ge06NFycI+Qs0vE"
    "crXwAHNg4kAs2QBAJGTod2x+nK5CIRYRK9crsFzKJEc31zotajq0WokiO90n3XKgYfOjAjTSyYEcszkS"
    "qW+OtbfieR8Q4t4geEVPKYxTjaKJAw0Mbu51RkR2qJDAPmXsj7Mgk177AS9Ia4loyWZjZmOI9yP2liQZ"
    "hSg7uAyFvepRVVZeTJS0+zfIxjUnZu1BDxIE4mbO1HE0qicDKSXfY2eL/OlZz4pTPtw6k0J45GlFv6ty"
    "1HMnAdtIcJfNppehLw7/SmuatQGc9tVldLOO3QtIvESRcd09vXT9y/IcSWVczKvJWxepAkvoW0Neg+NL"
    "Xkvf0Vcy0WqaiKqVHNzWYb0uN82tCObvQuQCOm6qWSKUGGS4q6pdC9XsnKZusda2lqPuSqqtnuLw3Aqi"
    "Xkd9VL7M0etnXh5ZyvrCuTucAOuq59gFyXkYuZjQktUcBY5rJ66lSJUoxOI0W/VdnPjgGTGYb3YwObD1"
    "2mRiZyaLSKcNFvM4sFh96kByu/SzA66N0P2EcEtoD5PJqe5YREWn9UwSrc8ai1K0aHYtLv2142VIXwFM"
    "lnUGAykgX6sK9e4F3+3GZ0GmD4AP0uSXls3vCKZqEZCzct4oWPDcvKl3jhtbxf/kxkWnGok0fbepkHoV"
    "dk2Q5TrVTrXwPQvqAPi8WPKZ1gxSUIEHsC/yx2x4ZAsbqGuvIkbZQxK55M7YVsHL2y+rl+0eRCkQ0KRV"
    "bumyMZFFh8PyM05h1db1owsI/IZs0tsDgYVUR+RAYcRAorPJRjteWS2gab1Xpc/SJhX1tVm3GYAmRwcD"
    "oRxY7OXg+9ZgU5g8ZdOaiT6P3Tp4VaqUKozpsPBpl5l2cs/ykQRNKhNSj6lbUmjpsfvM/Oaz6NN6ZBMA"
    "WUT+1R7i7ItjDZHJNfpjRscPDphtQGVQBXR5HXN6awAT6yjNN91gZ5E6jqrpoGnYqit4KvXwH7qGdpaj"
    "htwUWVdLCespO6J0UYz3Ii0X7auNSE3NFguLxgvfrprc6Ddj4y3mOxj26LJSKm5LpBa4zTi82aoZKjop"
    "befdk32I7QdpKJzSqeGimMyGu5Io1gn/9HaiNBZAQ/iosBAcF5SsnuRvOVtFOI4Spdx0cgVYAG3SQZaO"
    "2XDqWvkDdsxsu/CtRkueejYFuqKvedsAmbHsq+bSvUrrSInQ4Cw1d8kw9rn0hTcSSjda6EWSJu1qhDxk"
    "ZV7HotSYELCOJFV1ypz+v/FMigQ6nQ7KOGsHNKI2wsTijxgKxmgx+cLmemzkJ5NOCzbWThhbVmMNRUCy"
    "U09HZ0L3eBwND+MqcqpeiO5RVSl+avHnDAd1K3wbW75QBHtS6vg4M8ca4X6UzSXhUIbLRSfr6JAXWKwI"
    "RjpqWZsnKbINaTs9FdG8LImiRCJNAWczzdbGaZ+k8NL4DTpV1KQFObMeRN6F4ngplXWUI0VbOzsg9QOZ"
    "RWmpihMlUbYVisIUM84Ev69Ay4JB7NY4qlqiybF6Ye5mKcl2RTS6lzKVq+Nl6Cq6YmSL4Ps6zpFlAWH3"
    "ImWX9sySgoZ99RqauoPKGXUhzKNxOB60DQGv9KauV19AuqsslJ53Y1MyH73EMpUt4jqWUcB9WdQ/Dmu2"
    "wrtQ758doLA92Wnsgh2Rs9WX3E2wJHHLnXfqJpwiM8gWF9UeOqvvHjQB9KCzczRAM4FqSdXczf+i+VeJ"
    "Osd+9mR9gxSglutVD3zFGhKjpEq+26ZuNfw26zLGBh9qVAY9y7taVP+kV9eQtOzEm3ZnyiRyYSQQzSME"
    "V8E7ZCrW6IeZXwT9ekXa8YwDJcCRSXhN7DrI0MFrt1109BCJStZy9M7Q40OH1Vd7iLZbViejMbmiwsk2"
    "JJkxHOKI5dJl0403rbis6ncRAd4vC2nNFli3alMZM0lXO7LqjvcqyYbQKLGoY7F3B43DhPNwZe2KHdXk"
    "geZLj6P9kH2komCJYBP8Uq3hvZud1iH2y4vTaiWkCJ+5/tn+U8cAmFv/klhQxXtMpW2bE6hHAGRH2Yu6"
    "OmlKMQo+NEU7j4c0Wce5AszYD0vqtV1UEsOC51H/G3Au4KNCXGtEPbuFfsYDbTY0wNtMNgeotoclnpMT"
    "cNmklz4LkD6lAAftzvSZDZ3zMS840ZdJ6kalaXC2aJwuZb2vo1wm2XhtKhUw7iBypeLQcdNUVoNuNaCB"
    "T5YOkeFI+hXRt5lq9FKPCR4JKUoW+NOqqXcb4Tm7jXea2wGXb6i5ZZZyBUO+btsuiJ4AfIBwl8MZUj7Y"
    "3CmLXB7ltq34BIIM0ha4V29EHpeozDrqcPSULRWm7q1pp2ffoGqBR2v/eBMQdjfLUa2Zdv+BrDk1GSdP"
    "YUqrkg0pqc0yp5GOtn50k3YSBwSYFG2no1dP3SSgXB3fAGuaYbmSaHA+9J5syhHgYN7ludat1BbHwkZW"
    "5FZs0xW1eflgbYWn39ZRGIqXnMBYmmWCkU6xKlGK7EZrTHq8COHCqfInKuUvTJ0LnpFUNt8CY1puge2A"
    "5DijVddR2G0tpmVFPdU3fYZw6O01nxzfY31rOvoo2RnJLhDyTZCLZP2ElCKoBcFXtctToCb16KdnRbut"
    "IPSoLeCrG50UgLedjySZFdMB+uaxykAy+5BtOCnT2IGiI+r0024JYefS0uxRRlqH3tGpyvYVM6knkVzW"
    "bV1HEy6iC9VGbnqnw7Znux9ms2y2Gi20evD3hrV5Y+zmDlKMX71BozZbjDrtbLudWfIoZ9eIQIs5d5bN"
    "g+1E984mNayZcPJe6DMpJd54jmbH/QyVs163SSU7XroQdIWOxBbARzTGFQZuR1fGRilvamZylQo7E01U"
    "5VGibmGuxbfcVc5HCUG0ns46ofMocBebK/KFzYYk3SmM3Kt/8tqd/orJFEGz8LptA2K+Tb2S0iYBVjRu"
    "NC9SypeRuWhYsoNG2pi6nkk0VbCFpKMPd58lg/SaBVqkxjpZuBtPrt3nPDpEkShNPaYLlSNtSkb2VcRP"
    "AlDakJy7HF47Rx2IYMREHstY625LEKGegJ8oTzQ3PWJ0e2MFd7K0o4FGJNUY6JVbPgm2uskjEvpiJRg/"
    "xxJY6Rnd7nZ8Y3G+hBKRqSaCraZYeZMz1Yn6+dHnU2qRqqIf+QXbQUXoI4E3aiV2eVSAGD1goLyzXyVF"
    "Ea+1s3waqbZnAbF8J3Xn2OsguhR9aY3mtp928xs2LbIiILVDI4pdhzs65hF6CWA5/CtbomfHgWgkMI4b"
    "sS+kmRepE/q71nE48kMUUCNsDt1CG7tmRdZ57SS1xVCGaeEnmwjQ6M8IcuvOJOZuuje/SF2vBlCC2J7Q"
    "knaddrWeVy9WVyEIpzLW7HGlNU55t8J+1ana3fE6pmMJ0DC1MBBnJS+7iLYq8kDJsVkOWhJAmpsDZKJC"
    "YDbaXmdHYUcEPI8esTGpoB1NXF88WjCKtfu6F+aUWM+c7Upgk875pnUlO9iL5AlxU8GrPevIKgFxcXke"
    "Udbb1lFlZ1tEeVpRceEdV11WMRN7fWH17c/iW4BQAxFeZUO98pyjoUmLdC36h2jJfh1R0CwVmu0RrzTF"
    "HNO2+u4+VoKnNXYBTam7G6nenQMfknzhKxqtWxFFQJNe/VCtJYmy1CudnlgM+ugnT05duA4ivcUd0lnp"
    "/5MvG3DNg3oX5s+hcfqoKaryj1JE6cwODxR+YXVSlGGD+yV16TUOXReNKPOLbTRhp0yYBAtNN9n66bRF"
    "UKoJ8acD5i26PK/sJEtGs9LSZYJlDjzcie+xskNWp8MuGUDNWxjQMNHdK/kWB0w6wDY7EJRvYApnckK3"
    "o7b5mDNoB1PHmRU6vCI3xOgNUo3jikHaz/OQeHR29ACBWBSbDPZlIqcT5RCTeOmktohsBn0aR/dVd9ou"
    "LxxrzpBVbp1kwAAhXTFBlYGZZOI7VmXgZk5nKeM2LKdAM+I3xsoynIT06CAfP8m0LIz6IABejh91D/AC"
    "Bm73hMkyQS0XZKAB3DBpcD9Jvi+EcAC9z+6Yeaa4OiSYIoeBNxUjlvHZBujLAD9alMxMeAqTXgsLz6/j"
    "B07M/cCJHfQEcqgKyMTAYR54RYMGG5MU7wTEnTQbmBiZQfJ4cHqnI4gABBbB+YCFGItLuBZjVZkhix2O"
    "oeXuNsFhIy4auAsD92WimBwOPhG1iQKPcwHiQig9sHCLUGvgFk7ix7GcP81w++ERZXopX+e04piY7ZBo"
    "1s0Tty4n5V4s5sj7pm2CNvFHZo6zxgl3pM1cWzZJvjjB14HJSNNs+dxBrj/BnyZHZtFeQ53hLyfSPmCe"
    "T3LxgyQofh66bCYntw/f8FwqqZMcpOuMH6euZx6fZ2OezMThXijeVRWt4eDs50eon7Xf7xDpAWo1QToG"
    "Fus+OAwSV6z9AOebE7Oop3e+9yRTNIB7B07dUFFx7HCjx4ix4EzyRg1ToDe4zYEH6j1DrFgKA67QQlgH"
    "HIpF6WzDSx6x2s/aLFUnpIJJwmySOiZh6Nl5ntpRywTyixC8Az4PMquDrOSg1mVSXD/pvv2ajuxoYUbf"
    "AXIPhSbnc3S905ERwOGJLd6Qu8+BcsK1Q7aZPs59gTMsJGlBDRo0m5iQ4Ne1FNjKgiYW1DlRnBKeJ5QN"
    "BrdHHM95KmwEu6+RA7qdGi+gj+FwKpysyapMdSIRE9z7Sf+BsbTMKEg33+nn7j4CiMmbxH3Dakti1kn4"
    "NwG+RlZ+67ais6g7k94EWgzJo1/WhP41lwLAl+ORTER1QTOaTj8HLxkYg4HELMCNSQw/tb4YGAe/D+Ip"
    "mOLNieAoIxyfCZti0Ex4FufOdzKXiCLZpQGiMkhiTTqYDIfBVx8nHZraCe2sE3Jongq3fhDATqLNCegw"
    "yVwMQsVJcfwkYp+Ueg0Wb0Ck6jChxlKRIhlQkBuSOIh8x1LlVBQ1pk0LR6X00ihhJjEIC8h1Ihp2mHai"
    "+YAEpAfCsSLuWcQI2LyFoRnwx6aBFVDlwiWcVEtNGj4PciETPylkzYhnqN5xqHCQJtmZmNNOScAczqR8"
    "9qNLwHL+oskwJ7BRpdSdYU16RtKtpowk/rRIq+nFIAtNEzB0HdiQ0E5ra4hZr8MHSETYAxx3TjWRo+Yx"
    "IXosIO9hA5OTzw/FBbd/8vydRpU63XoJE/szCDTXpTO41DmNI45+9WYRG5+AcoNB3qEntT5+FXxrj+WE"
    "1TIhyo7sCl+H/k9OiOc061zAf1609RtkNydsYd2QBQdrkGRYEIp1CKjYvgWH08nN0lxkGZdkfiDpZLIn"
    "BdujaNm5V7COBVal9eq6N1hOcJMBY3KSH+g0Jx5E5BMkOnSU4QaEppjAxzKMpZOpptORQPfGXAeyGPbx"
    "GWLIAhQuyhyblzydwHjpc069Mb928epcYWdLAOxITTAVPqBFxaFc1vqzQQ7xJcs3IwMKVhYDgprxDVtD"
    "m5yrvaY6WFlXixNiLjJVJ5Ze6VkkEpk0blikhZ1OYB8XmfoVzd7b0VaFhhQd2DzFxyi0DL4k+CTRTZK5"
    "SmRqsWuySyFd4jvGLiWbTxeo688qDoLXW7CRBE+jPnRVlFGdapdFhr5gVAiASK6Mrj1B8RIyp3JQQdbt"
    "vPzjr7/82y//8D/qr7e48V/9df767f+/vRq/3l/z7f/v199+X/w5v/3eH/XX9vz2Plv8kY/yEf70/Hk8"
    "n/52veR1km9/LlNe76zcBf/Hnwev5+sdc1+1vvnftw90v8fHqfEVzz1X//Xtrz2+psaDfHtWFuLb/zpv"
    "4cPP2/p+c5pxmRy/nMdNpNf78q/53U3W4zPf/rv2U79+l8/VeK5xeQ9v/vjugsfT1TeLNd9c8e3970V5"
    "f3P1zV/mvrf42/70awniF/PdCu3/2z+OX59/ef1uvl2P/V3HE9RbdF4PclwtHe9I85NlPb/k7dfkQwbf"
    "fe4lCWm+X+w3IvfxiT75no8S9OZjx9XPV2/+Md9+xzvZ+Lj2b5dyvvb0/S2ON1d4K431zfe/9MRxBE45"
    "+fA82S8Yb5f67T7ND/f85r/5YSHyt9M7Pzzk2219e1OfXPfNcn/2JR9u7/1zHk+yl669PaNfPNd898f6"
    "/fudn2z754/0/ph/V7a2IpqHBNXPD9InX/PxvutnK/jpf4+GTfPTJ4476IcAfKYPvt6eD+s63wrSJyv9"
    "UbF+snqvv5X3Eve9h/146fH5XcyP61u/XM76Y9Gpn79+r9Tevzf+9MmhqD88VR9//ca+fL4Pnx69+n75"
    "PtxIVRnk8yR+sHBv5f3r7/ns2PxYsvJ3hL5+lNLuKuR3d71P4g9O+Qdr/aW2/OyP+aOify918+sD+bV9"
    "/nIBvzzfXxyN+l4/fn0Gv9yY8tkxen9cvvI9PnniNyb/3V7X70lwKt9R1fMHuvIQl/wTUvhOGV7fVRqf"
    "WrP6mcb5+RPxcWHmd3TXd47h97b4vVo4fn4mut+3RPMHQl5fHucX2vRHt/rxbuZ3nvqrG+s/dF8+avf8"
    "4UB95xNvTvQHo/rdfbl+Zg2+lNz65T7Vj2Ywf6Hz6o8VzMdX453vWH5oVD9R+O+kuPxA2OrPiU79wV8/"
    "ecT6U3a4/Jxz9ong1B8ene/sz8/Jw8f/rs8eun/ma9UvVqH+nWp3fvnYP1Yn9ec07o8/9RPq/itBr99V"
    "1fXnPbcfit73VPgnXsH1u/y+n9GBP9qT/DY0/vC38h019HeK8xcbN39aK/7sf/N37d5HRKH/yDn73hW+"
    "Ui9//4rVn7zKV5qv/qSCqb/LZn8lBv2zN9TfdaQ+aLL5exZs/NQh/vy4/8wN9nc/f+dupPXD/f3q2fuP"
    "H6v86ADV3+lU/qyn/F/WBD95L+OzG+l/p0X9wWrNH1yqfi8qT+UDEvsjkf704cv3tXv9/WbqI/z6+49q"
    "+S/5MOUrK/UTRvg7fsTH3X1ztf7mDQdQ1V8hqUkYXvaftTb1pw5W/8nV+eRTP2WIvgwCyruNLe/U4Pfj"
    "jPw7lfebI9S/B4WfUGv/PXqn/tTd/YTfUD6ACp+hID/5X/n8YNSfcBvrD65aXyfly5XqP6vnSoAv9a1s"
    "fRCP+uNne3P//d0B+3CFcohi/8Lm18+/67v38ibIKu+fvH969j4AAp+fs/UDj6y8EfL6qUGv55O/3c/y"
    "Ha/+0MzP0r6Vh/odNIu/9I/HobwD4ELlfU+nl89V5v592Vr82ONv/LVTq7IR5e3ulfcfUTI+Wbn30GV5"
    "Vj32ltd9f0U5ZfHD1/TXe/prF/yvv/b03e0e91/Pbznu/bjo9Tpn795atmHxe47fzpCmye2Xz2+je+V4"
    "bm+376d7J/d+uW9q3t7zt/Ftm95e5+LyHTLAt3+Md3f6erYP2dT6LskwPyjY02B2E/P9gztdP+Kvb7Hf"
    "z7KR5818iU5Fzj1+P95mRtJ59H5kD+enmYV6kCxOfOZ9PuBDCv5N+uGrH3Gx+WGNXk9d3+UPPuAN87Oo"
    "oPxk9D0/Eig+Jpbmp7SD3+fXfc+1+5lAon+WzPtJN3V+Dx549/b+nbTZzy3B74tvv7ilT/Dj8sMc1Pyx"
    "Y/Q7fJ8vPJLr6+uVz92Nj1hv+X4M0T5HDMsJab9IUfUn0ij9dAfnB8/mS/gql19zfv57/ePdv//u//6O"
    "i9SfveC9R/+ly9a/9x5+76fqlwuSjj/+cKmun/u29OEN1/Pv138f766+/cfHP7274A/Xsnx9kXe/KZ9d"
    "9vzHtV+mr26mfvjsV3fYv/7Ux5fvlr0eN1w+W/Dv3Fv67jd+9Qgf31A/vOE77++fvCGdlypvr3x+vH93"
    "Gd89UXv7ReUToVrxi+vDApw3+/FG2mf3mL79t54/pnhXe3PT6bvHqX92pj4+bP9iQ8/P1pCB8tMn9zuH"
    "qP7oH59+pH62+28e9/D589v/fy9GYQFfH64qqhz+Pe/tz1vL9tEHX5NuX1yPPB+/f3z4/vrfeP3627qV"
    "WL1znduH52KT6z/+f/8/iOVTHQEQAQA="
)

# ---------------------------------------------------------------- Win32

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11
MOD_ALT = 0x1
MOD_CONTROL = 0x2
WM_HOTKEY = 0x0312
VK_OEM_PLUS = 0xBB
VK_OEM_MINUS = 0xBD

KEY = (1, 2, 3)            # 透明色键
PANEL = (13, 18, 26)       # 面板底(视频取色近似 #0d1219)
PANEL2 = (7, 10, 15)       # 控制台底
BORDER = (38, 48, 62)
TITLE_C = (210, 220, 232)
DIM_C = (110, 124, 140)
ACCENT = (232, 150, 60)
AMBER_HI = (255, 196, 110)
CYAN = (80, 200, 190)
RED_C = (224, 96, 96)
GREEN = (60, 200, 110)
WHITE = (232, 240, 250)
CHIP_BG = (34, 44, 58)

DEFAULT_CONFIG = {
    "x": None, "y": 100, "scale": 1.0, "clickthrough": True,
    "hide": False, "rotate_deg_per_frame": 0.04,   # 视频里脑云近乎静置
    "show": {"brain": True, "stats": True, "spark": True, "console": True},
    "exclude_from_capture": True,
    # 自定义监视键:vk=虚拟键码,label=键帽,desc=动作,groups=激发分群
    "watch_keys": [
        {"vk": 0x57, "label": "W", "desc": "向前移动",
         "groups": ["vnc_motor", "descending_neuron"]},
        {"vk": 0x41, "label": "A", "desc": "向左移动", "groups": ["vnc_motor"]},
        {"vk": 0x53, "label": "S", "desc": "向后移动", "groups": ["vnc_motor"]},
        {"vk": 0x44, "label": "D", "desc": "向右移动", "groups": ["vnc_motor"]},
        {"vk": 0xA0, "label": "SHIFT", "desc": "冲刺",
         "groups": ["vnc_motor", "vnc_efferent"]},
        {"vk": 0x20, "label": "SPACE", "desc": "跳跃",
         "groups": ["vnc_efferent", "vnc_motor"]},
        {"vk": 0x01, "label": "M1", "desc": "普通攻击", "groups": ["cb_motor"]},
        {"vk": 0x02, "label": "M2", "desc": "瞄准",
         "groups": ["cb_sensory", "visual_projection"]},
    ],
    "cap_keys": ["W", "A", "S", "D", "SHIFT", "SPACE"],   # 键帽行显示顺序
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("msyhbd.ttc" if bold else "msyh.ttc"), "msyh.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:\\Windows\\Fonts\\{name}", size)
        except Exception:
            continue
    return ImageFont.load_default()


class InputMirror:
    """真实输入镜像:按键状态 + 鼠标速度 + 事件流(供控制台)。"""

    def __init__(self, watch: list[dict]):
        self.user32 = ctypes.windll.user32
        self.watch = watch
        self._prev_cur = None
        self.events: deque[str] = deque(maxlen=9)
        self._last_mouse_log = 0.0
        # 每秒窗统计
        self.win_t0 = time.perf_counter()
        self.win_events = 0
        self.win_dx = 0
        self.win_dy = 0
        self.rate = 0.0          # 事件/秒(上一秒)
        self.speed = 0.0         # 鼠标 px/s(上一秒)

    def poll(self):
        held = [k for k in self.watch
                if self.user32.GetAsyncKeyState(int(k["vk"])) & 0x8000]

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        p = POINT()
        dx = dy = 0
        if self.user32.GetCursorPos(ctypes.byref(p)):
            cur = (p.x, p.y)
            if self._prev_cur is not None:
                dx, dy = cur[0] - self._prev_cur[0], cur[1] - self._prev_cur[1]
            self._prev_cur = cur
        self.win_dx += dx
        self.win_dy += dy
        # 事件流:mouse_move 节流 160ms
        now = time.perf_counter()
        if (dx or dy) and now - self._last_mouse_log > 0.16:
            self._last_mouse_log = now
            self.log_event("mouse_move", f'"dx":{dx},"dy":{dy}')
            self.win_events += 1
        # 每秒结算
        if now - self.win_t0 >= 1.0:
            self.rate = self.win_events / (now - self.win_t0)
            self.speed = math.hypot(self.win_dx, self.win_dy) / (now - self.win_t0)
            self.win_t0, self.win_events = now, 0
            self.win_dx, self.win_dy = 0, 0
        return held, (dx, dy)

    def log_event(self, kind: str, payload: str = ""):
        t = time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time()*1000)%1000:03d}"
        line = f'{{"ts":"{t}","type":"{kind}"'
        if payload:
            line += f",{payload}"
        self.events.append(line + "}")

    def key_down(self, k):
        self.log_event("key_down", f'"key":"{k["label"]}","desc":"{k["desc"]}"')
        self.win_events += 1

    def key_up(self, k):
        self.log_event("key_up", f'"key":"{k["label"]}"')
        self.win_events += 1


class HotkeyThread(threading.Thread):
    def __init__(self, cmd_q: "queue.Queue"):
        super().__init__(daemon=True, name="overlay-hotkey")
        self.q = cmd_q
        self.keys = [
            (MOD_CONTROL | MOD_ALT, 0x25),          # 0 ←
            (MOD_CONTROL | MOD_ALT, 0x27),          # 1 →
            (MOD_CONTROL | MOD_ALT, 0x26),          # 2 ↑
            (MOD_CONTROL | MOD_ALT, 0x28),          # 3 ↓
            (MOD_CONTROL | MOD_ALT, VK_OEM_PLUS),   # 4 放大
            (MOD_CONTROL | MOD_ALT, VK_OEM_MINUS),  # 5 缩小
            (MOD_CONTROL | MOD_ALT, 0x43),          # 6 C 穿透
            (MOD_CONTROL | MOD_ALT, 0x42),          # 7 B 脑图
            (MOD_CONTROL | MOD_ALT, 0x54),          # 8 T 统计格
            (MOD_CONTROL | MOD_ALT, 0x4B),          # 9 K 波形
            (MOD_CONTROL | MOD_ALT, 0x4A),          # 10 J 控制台
            (MOD_CONTROL | MOD_ALT, 0x58),          # 11 X 捕获排除
            (MOD_CONTROL | MOD_ALT, 0x48),          # 12 H 隐藏
            (MOD_CONTROL | MOD_ALT, 0x52),          # 13 R 重载配置
            (MOD_CONTROL | MOD_ALT, 0x51),          # 14 Q 退出
            (MOD_CONTROL | MOD_ALT, 0x44),          # 15 D 调整模式(拖动/缩放)
        ]

    def run(self):
        user32 = ctypes.windll.user32
        for i, (mod, vk) in enumerate(self.keys, 1):
            user32.RegisterHotKey(None, i, mod | 0x4000, vk)
        self.tid = ctypes.windll.kernel32.GetCurrentThreadId()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self.q.put(int(msg.wParam) - 1)
        for i in range(1, len(self.keys) + 1):
            user32.UnregisterHotKey(None, i)

    def stop(self):
        ctypes.windll.kernel32.PostThreadMessageW(getattr(self, "tid", 0), 0x0012, 0, 0)


class Overlay:
    def __init__(self, args):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.cfg = load_config()
        if args.reset:
            self.cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        self.scale = float(args.scale) if args.scale else float(self.cfg.get("scale", 1.0))
        self.show = self.cfg.get("show", dict(DEFAULT_CONFIG["show"]))
        self.clickthrough = bool(self.cfg.get("clickthrough", True))
        self.exclude = bool(self.cfg.get("exclude_from_capture", True))
        self.hide = bool(self.cfg.get("hide", False))
        self.rot = float(self.cfg.get("rotate_deg_per_frame", 0.25)) * math.pi / 180.0
        self.fps = args.fps
        self.state_path = args.state
        self.adjust = False          # 调整模式:可拖动/缩放(点击穿透临时关闭)
        self._locked_clickthrough = self.clickthrough

        self.f_title = _load_font(int(12 * self.scale), True)
        self.f_lab = _load_font(int(9 * self.scale))
        self.f_val = _load_font(int(12 * self.scale), True)
        self.f_sm = _load_font(int(10 * self.scale))
        self.f_con = _load_font(int(9 * self.scale))
        self.f_cap = _load_font(int(10 * self.scale), True)
        self.f_foot = _load_font(int(9 * self.scale))

        # ---- 点云(布局:脚本模式用仓库文件;exe 用内嵌数据)
        lay = self._load_layout()
        self.lx = np.asarray(lay["x"], np.float32) - 0.5
        self.ly = np.asarray(lay["y"], np.float32) - 0.5
        self.groups = np.asarray(lay["group"])
        rng = np.random.default_rng(20261005)
        ug = {g: i for i, g in enumerate(sorted(set(self.groups)))}
        self.g_idx = {g: np.flatnonzero(self.groups == g) for g in ug}
        gz = np.array([((ug[g] * 0.618) % 1.0) - 0.5 for g in self.groups], np.float32)
        self.lz = np.clip(gz * 0.55 + rng.standard_normal(len(self.groups)) * 0.07,
                          -0.42, 0.42).astype(np.float32)
        self.a = np.full(len(self.groups), 0.22, np.float32)
        self.theta = 0.0
        self.spark = deque([0.2] * 110, maxlen=110)

        # ---- 输入 / 统计
        self._reload_watch()
        self.mirror = InputMirror(self.watch)
        self.held_prev: set[str] = set()
        self.total_events = 0
        self.frames = 0
        self.t_start = time.perf_counter()

        # ---- 热键
        self.cmd_q: "queue.Queue" = queue.Queue()
        self.hotkeys = HotkeyThread(self.cmd_q)
        self.hotkeys.start()

        # ---- 窗口
        self.W, self.H = int(424 * self.scale), int(636 * self.scale)
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.config(bg="#%02x%02x%02x" % KEY)
        self.root.attributes("-transparentcolor", "#%02x%02x%02x" % KEY)
        x = self.cfg.get("x")
        y = self.cfg.get("y", 100)
        if x is None:
            x = self.root.winfo_screenwidth() - self.W - 48
        self.root.geometry(f"{self.W}x{self.H}+{int(x)}+{int(y)}")
        self.label = tk.Label(self.root, bd=0, bg="#%02x%02x%02x" % KEY)
        self.label.pack()
        self.root.update_idletasks()
        self.root.update()
        self.root.attributes("-topmost", True)
        self._hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        self._apply_exstyle()
        self._apply_capture_affinity()
        self.root.after(1000, self._reassert_style)
        self.root.after(int(1000 / self.fps), self._tick)
        if self.hide:
            self.root.withdraw()

    # ------------------------------------------------------------ 窗口属性

    @staticmethod
    def _load_layout() -> dict:
        """脑布局来源:脚本模式优先仓库文件(便于自定义);exe 用内嵌数据。"""
        if not IS_FROZEN:
            p = ROOT / ".cache" / "ann_layout.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                return {"x": d["x"], "y": d["y"], "group": d["group"]}
        slim = json.loads(gzip.decompress(
            base64.b64decode("".join(_EMBEDDED_LAYOUT_B64))))
        groups = slim["groups"]
        return {"x": slim["x"], "y": slim["y"],
                "group": [groups[i] for i in slim["g"]]}

    def _apply_exstyle(self):
        user32 = ctypes.windll.user32
        ex = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
        ex |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST
        ex = (ex | WS_EX_TRANSPARENT) if self.clickthrough else (ex & ~WS_EX_TRANSPARENT)
        user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, ex)
        user32.SetWindowPos(self._hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)

    def _apply_capture_affinity(self):
        user32 = ctypes.windll.user32
        aff = WDA_EXCLUDEFROMCAPTURE if self.exclude else WDA_NONE
        if not user32.SetWindowDisplayAffinity(self._hwnd, aff):
            print("[overlay] SetWindowDisplayAffinity 失败(需 Win10 2004+)", flush=True)

    def _reassert_style(self):
        self._apply_exstyle()
        self._apply_capture_affinity()

    def _save_geometry(self):
        # clickthrough 保存"锁定值"而非调整模式的临时值 —— 否则调整中途
        # 崩溃/退出会把 False 毒化进配置,下次启动永久失去穿透
        self.cfg.update(x=self.root.winfo_x(), y=self.root.winfo_y(),
                        scale=self.scale,
                        clickthrough=getattr(self, "_locked_clickthrough",
                                             self.clickthrough),
                        show=self.show, exclude_from_capture=self.exclude,
                        hide=self.hide)
        save_config(self.cfg)

    def _reload_watch(self):
        cfg = load_config()
        self.watch = cfg.get("watch_keys", DEFAULT_CONFIG["watch_keys"])
        self.show = cfg.get("show", self.show)
        self.cap_keys = cfg.get("cap_keys", DEFAULT_CONFIG["cap_keys"])
        if getattr(self, "mirror", None) is not None:
            self.mirror.watch = self.watch

    # ------------------------------------------------------------ 热键

    def _handle_cmd(self, idx: int):
        s = 24
        act = {0: lambda: self._move(-s, 0), 1: lambda: self._move(s, 0),
               2: lambda: self._move(0, -s), 3: lambda: self._move(0, s),
               4: lambda: self._rescale(1.1), 5: lambda: self._rescale(0.9),
               6: self._toggle_click, 7: lambda: self._tog("brain"),
               8: lambda: self._tog("stats"), 9: lambda: self._tog("spark"),
               10: lambda: self._tog("console"), 11: self._toggle_excl,
               12: self._toggle_hide, 13: self._reload_watch,
               14: self._quit, 15: self._toggle_adjust}
        fn = act.get(idx)
        if fn:
            fn()
        if idx != 14:
            self._save_geometry()

    def _toggle_click(self):
        self.clickthrough = not self.clickthrough
        self._locked_clickthrough = self.clickthrough
        self._apply_exstyle()

    def _toggle_excl(self):
        self.exclude = not self.exclude
        self._apply_capture_affinity()

    def _toggle_hide(self):
        self.hide = not self.hide
        self.root.withdraw() if self.hide else self.root.deiconify()

    def _toggle_adjust(self):
        """调整模式:临时关穿透 → 鼠标拖动移动、滚轮缩放;再按一次恢复。"""
        self.adjust = not self.adjust
        if self.adjust:
            self._prev_clickthrough = self.clickthrough
            self.clickthrough = False
            self._apply_exstyle()
            self.root.bind("<ButtonPress-1>", self._on_press, add=True)
            self.root.bind("<B1-Motion>", self._on_drag, add=True)
            self.root.bind("<MouseWheel>", self._on_wheel, add=True)
        else:
            self.root.unbind("<ButtonPress-1>")
            self.root.unbind("<B1-Motion>")
            self.root.unbind("<MouseWheel>")
            self.clickthrough = getattr(self, "_prev_clickthrough", False)
            self._locked_clickthrough = self.clickthrough
            self._apply_exstyle()
            self._save_geometry()

    def _on_press(self, e):
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _on_drag(self, e):
        self.root.geometry(f"+{e.x_root - self._drag[0]}+{e.y_root - self._drag[1]}")

    def _on_wheel(self, e):
        self._rescale(1.06 if e.delta > 0 else 0.94)

    def _tog(self, key):
        self.show[key] = not self.show.get(key, True)

    def _move(self, dx, dy):
        self.root.geometry(f"+{self.root.winfo_x() + dx}+{self.root.winfo_y() + dy}")

    def _rescale(self, factor):
        self.scale = float(np.clip(self.scale * factor, 0.6, 2.2))
        # 必须同步 W/H:渲染坐标全部由它们推导,不同步会让 y0>y1 崩溃
        self.W, self.H = int(424 * self.scale), int(636 * self.scale)
        self.root.geometry(f"{self.W}x{self.H}")
        self.f_title = _load_font(int(12 * self.scale), True)
        self.f_lab = _load_font(int(9 * self.scale))
        self.f_val = _load_font(int(12 * self.scale), True)
        self.f_sm = _load_font(int(10 * self.scale))
        self.f_con = _load_font(int(9 * self.scale))
        self.f_cap = _load_font(int(10 * self.scale), True)
        self.f_foot = _load_font(int(9 * self.scale))

    def _quit(self):
        self._save_geometry()
        self.hotkeys.stop()
        self.root.destroy()

    # ------------------------------------------------------------ 脑活动

    def _brain_activity(self):
        """真实输入 → 分群激发;--state 新鲜时优先真实训练活动度。

        返回 (活动度, held, 说明, (mdx, mdy))。
        """
        held, (mdx, mdy) = self.mirror.poll()
        excl = " · 已排除捕获" if self.exclude else ""
        if self.state_path:
            p = ROOT / self.state_path
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                if time.time() - p.stat().st_mtime < 5.0 and s.get("activity"):
                    a = np.asarray(s["activity"], np.float32)
                    if a.size == len(self.a):
                        return a, held, "实时训练状态" + excl, (mdx, mdy)
            except Exception:
                pass
        target = 0.20 + 0.06 * np.sin(
            np.arange(len(self.a)) * 0.013 + time.perf_counter() * 0.4)
        for k in held:
            for g in k.get("groups", []):
                idx = self.g_idx.get(g)
                if idx is not None:
                    target[idx] = np.minimum(1.0, target[idx] + 0.45)
        mspeed = abs(mdx) + abs(mdy)
        if mspeed:
            for g in ("ol_sensory", "ol_intrinsic", "visual_projection"):
                idx = self.g_idx.get(g)
                if idx is not None:
                    target[idx] = np.minimum(1.0, target[idx] + min(0.5, mspeed * 0.02))
        self.a = 0.80 * self.a + 0.20 * (target + 0.05 * np.random.default_rng(
            1).standard_normal(len(self.a))).astype(np.float32)
        return self.a, held, "输入映射 · 非真实仿真" + excl, (mdx, mdy)

    # ------------------------------------------------------------ 渲染部件

    def _frame_pil(self):
        """PIL 侧:面板底 + 标题栏 + 状态行。"""
        from PIL import ImageDraw

        img = self.Image.new("RGB", (self.W, self.H), KEY)
        d = ImageDraw.Draw(img)
        s = self.scale
        m = int(8 * s)
        d.rectangle([m, m, self.W - m, self.H - m], fill=PANEL,
                    outline=(ACCENT if self.adjust else BORDER),
                    width=(2 if self.adjust else 1))
        d.line([m, int(26 * s), self.W - m, int(26 * s)], fill=BORDER, width=1)
        if self.adjust:
            d.text((m + 10 * s, int(6 * s)),
                   "调整模式 · 拖动移动 / 滚轮缩放 / Ctrl+Alt+D 完成",
                   font=self.f_title, fill=ACCENT)
        else:
            d.text((m + 10 * s, int(6 * s)), "FLY / 神经活动", font=self.f_title,
                   fill=TITLE_C)
        right = "已排除捕获" if self.exclude else "捕获可见"
        d.text((self.W - m - 10 * s, int(7 * s)), right, font=self.f_foot,
               fill=(GREEN if self.exclude else RED_C), anchor="ra")
        pulse = 120 + int(80 * math.sin(time.perf_counter() * 3))
        d.ellipse([m + 10 * s, int(32 * s), m + 17 * s, int(39 * s)],
                  fill=(pulse, GREEN[1], GREEN[2]))
        d.text((m + 22 * s, int(31 * s)), "神经活动报告中", font=self.f_sm,
               fill=TITLE_C)
        return img, d, m, s

    def _brain(self, arr, act, box):
        """numpy 侧:脑点云(琥珀主色,静置微旋)。box=(x0,y0,x1,y1)。"""
        s = self.scale
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + int(6 * s)
        bw, bh = int((x1 - x0) * 0.94), int((y1 - y0) * 0.88)
        self.theta = (self.theta + self.rot) % (2 * math.pi)
        c, sn = math.cos(self.theta), math.sin(self.theta)
        x2 = self.lx * c + self.lz * sn
        z2 = -self.lx * sn + self.lz * c
        dpt = 1.0 / (1.0 - 0.38 * z2)
        px = (cx + x2 * bw * dpt).astype(np.int32)
        py = (cy + self.ly * bh * dpt).astype(np.int32)
        ok = (px >= x0 + 2) & (px < x1 - 2) & (py >= y0 + 2) & (py < y1 - 2)
        px, py, a, dpt = px[ok], py[ok], act[ok], dpt[ok]
        # 视频观感:静息也有清晰的琥珀色,活跃处烧成亮琥珀
        hot = np.clip((a - 0.10) * 1.45, 0.0, 1.0) ** 0.8
        br = np.clip(0.62 + 0.38 * hot, 0, 1) * np.clip(0.55 + 0.45 * dpt, 0.4, 1.25)
        col = np.empty((len(px), 3), np.float32)
        for k in range(3):   # 暗琥珀 (128,98,60) -> 亮琥珀 AMBER_HI
            col[:, k] = (128 + (AMBER_HI[k] - 128) * hot) * br
        col = col.astype(np.uint8)
        big = a > 0.78
        mid = (a > 0.52) & ~big
        small = ~big & ~mid
        Hh, Ww = arr.shape[:2]
        for m_, sel in ((0, small), (1, mid), (2, big)):
            if not sel.any():
                continue
            xs, ys, cs = px[sel], py[sel], col[sel]
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if m_ == 0 and (ox or oy):
                        continue
                    if m_ == 1 and abs(ox) + abs(oy) > 1:
                        continue
                    arr[np.clip(ys + oy, 0, Hh - 1),
                        np.clip(xs + ox, 0, Ww - 1)] = cs

    def _fly(self, d, cx, cy, size, last_dx):
        """PIL 侧:果蝇插画(头朝上、红复眼、翅后掠)+ 指示箭头。"""
        s = self.scale
        u = size / 100.0
        # 翅膀(半透明浅灰蓝,从胸向后上方掠出,先画在底层)
        for sx in (-1, 1):
            d.polygon([(cx + sx * 6 * u, cy - 16 * u),
                       (cx + sx * 58 * u, cy - 54 * u),
                       (cx + sx * 62 * u, cy - 34 * u),
                       (cx + sx * 22 * u, cy + 4 * u)],
                      fill=(148, 172, 198), outline=(196, 214, 232))
        # 腿(每侧 3 条,细)
        for sx in (-1, 1):
            for i, (x2, y2) in enumerate(((26, 2), (30, 16), (24, 30))):
                d.line([cx + sx * 8 * u, cy - 12 * u + i * 12 * u,
                        cx + sx * x2 * u, cy + y2 * u],
                       fill=(96, 66, 28), width=max(1, int(1.5 * u)))
        # 腹部(深琥珀带条纹)
        d.ellipse([cx - 9 * u, cy + 0 * u, cx + 9 * u, cy + 46 * u],
                  fill=(168, 120, 54), outline=(104, 72, 30))
        for yy in (10, 20, 30):
            d.line([cx - 8 * u, cy + yy * u, cx + 8 * u, cy + yy * u],
                   fill=(112, 78, 34), width=max(1, int(2 * u)))
        # 胸
        d.ellipse([cx - 11 * u, cy - 24 * u, cx + 11 * u, cy + 4 * u],
                  fill=(198, 148, 72), outline=(104, 72, 30))
        # 头 + 大红复眼(占头大半,视频里最醒目的特征)
        d.ellipse([cx - 9 * u, cy - 38 * u, cx + 9 * u, cy - 22 * u],
                  fill=(190, 142, 66), outline=(104, 72, 30))
        d.ellipse([cx - 8 * u, cy - 36 * u, cx - 0.5 * u, cy - 24 * u],
                  fill=(200, 42, 40))
        d.ellipse([cx + 0.5 * u, cy - 36 * u, cx + 8 * u, cy - 24 * u],
                  fill=(218, 56, 46))
        # 指示箭头(随最近鼠标水平方向,视频里的橙色弧箭头)
        ax = cx + int(66 * u)
        ay = cy - int(30 * u)
        if last_dx != 0:
            sx = 1 if last_dx > 0 else -1
            d.line([ax - sx * 18 * u, ay + 8 * u, ax + sx * 12 * u, ay - 4 * u],
                   fill=ACCENT, width=max(2, int(3.4 * u)))
            d.polygon([(ax + sx * 20 * u, ay - 8 * u),
                       (ax + sx * 8 * u, ay - 14 * u),
                       (ax + sx * 14 * u, ay + 2 * u)], fill=ACCENT)

    def _keycaps(self, d, x0, y0, held_labels, width):
        """PIL 侧:键帽行(按下的点亮),撑满给定宽度。"""
        s = self.scale
        n = max(1, len(self.cap_keys))
        gap = int(5 * s)
        cw = (width - gap * (n - 1)) / n
        ch = int(28 * s)
        x = x0
        for name in self.cap_keys:
            on = name in held_labels
            d.rounded_rectangle([x, y0, x + cw, y0 + ch], radius=int(4 * s),
                                fill=(ACCENT if on else CHIP_BG),
                                outline=(ACCENT if on else BORDER), width=1)
            d.text((x + cw / 2, y0 + ch / 2), name, font=self.f_cap,
                   fill=(30, 20, 8) if on else DIM_C, anchor="mm")
            x += cw + gap

    def _stat_cell(self, d, x0, y0, w, label, val, frac, color):
        s = self.scale
        d.text((x0, y0), label, font=self.f_lab, fill=DIM_C)
        d.text((x0 + w - 4 * s, y0 - int(1 * s)), val, font=self.f_sm,
               fill=TITLE_C, anchor="ra")
        bx0, bx1 = x0, x0 + w - int(6 * s)
        by = y0 + int(15 * s)
        d.rectangle([bx0, by, bx1, by + int(4 * s)], fill=(24, 32, 42))
        fw = int((bx1 - bx0) * float(np.clip(frac, 0, 1)))
        if fw > 0:
            d.rectangle([bx0, by, bx0 + fw, by + int(4 * s)], fill=color)

    def _stats_grid(self, d, x0, y0, act, held, note):
        """PIL 侧:3×2 分组数据格(视频里的表格)。"""
        s = self.scale
        w = (self.W - 2 * x0 - int(16 * s)) / 3
        groups = {
            "视叶": ("ol_sensory", "ol_intrinsic", "visual_projection"),
            "中央脑": ("cb_intrinsic", "cb_sensory", "cb_motor"),
            "腹神经索": ("vnc_intrinsic", "vnc_motor", "vnc_efferent"),
        }
        cells = []
        for i, (lab, gs) in enumerate(groups.items()):
            idx = np.concatenate([self.g_idx[g] for g in gs if g in self.g_idx])
            m = float(np.mean(act[idx])) if idx.size else 0.0
            cells.append((lab, f"{m*100:.0f}%", m, ACCENT))
        cells += [("鼠标", f"{self.mirror.speed:.0f}px/s",
                   min(1.0, self.mirror.speed / 800), CYAN),
                  ("按键", f"{len(held)}", min(1.0, len(held) / 3), GREEN),
                  ("事件", f"{self.mirror.rate:.0f}/s",
                   min(1.0, self.mirror.rate / 20), TITLE_C)]
        for i, (lab, val, frac, col) in enumerate(cells):
            cx = x0 + (i % 3) * (w + int(8 * s))
            cy = y0 + (i // 3) * int(36 * s)
            self._stat_cell(d, cx, cy, w, lab, val, frac, col)

    def _spark(self, d, x0, y0, x1, y1):
        """PIL 侧:底部全宽波形条(活动度,青色细线)。"""
        d.rectangle([x0, y0, x1, y1], fill=PANEL2, outline=BORDER)
        vals = np.asarray(self.spark, np.float32)
        hi = max(0.35, float(vals.max()) * 1.15)
        pts = [(x0 + (x1 - x0) * i / (len(vals) - 1),
                y1 - (y1 - y0 - 2) * float(v) / hi - 1) for i, v in enumerate(vals)]
        d.line(pts, fill=CYAN, width=1)
        d.text((x1 - 5, y0 + 2), "60s", font=self.f_lab, fill=DIM_C, anchor="ra")

    def _console(self, d, x0, y0, x1, y1):
        """PIL 侧:JSON 事件控制台(真实输入事件流)。"""
        s = self.scale
        d.rectangle([x0, y0, x1, y1], fill=PANEL2, outline=BORDER)
        d.text((x0 + 6, y0 + 2), "input_events", font=self.f_lab, fill=DIM_C)
        n = max(3, int((y1 - y0 - int(18 * s)) / (11 * s)))
        evs = list(self.mirror.events)[-n:]
        yy = y0 + int(16 * s)
        for line in evs:
            if yy > y1 - int(12 * s):
                break
            d.text((x0 + 6, yy), line, font=self.f_con, fill=(150, 165, 182))
            yy += int(11 * s)

    # ------------------------------------------------------------ 主循环

    def _tick(self):
        # 整帧 try/finally:渲染出错也必须继续调度 —— 否则热键队列没人处理,
        # 调整模式会永久卡死(实测教训)。
        t0 = time.perf_counter()
        err = None
        try:
            self._tick_body(t0)
        except Exception:
            import traceback
            err = traceback.format_exc()
        if err is not None and not getattr(self, "_err_logged", False):
            self._err_logged = True
            try:
                (CONFIG_PATH.parent / "overlay_render.log").write_text(
                    err, encoding="utf-8")
            except Exception:
                pass
            if not IS_FROZEN:
                print(err, file=sys.stderr)
        if self.root.winfo_exists():
            self.root.after(max(4, int(1000 / self.fps - (time.perf_counter() - t0) * 1000)),
                            self._tick)

    def _tick_body(self, t0):
        from PIL import ImageDraw

        while True:
            try:
                idx = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            self._handle_cmd(idx)
            if not self.root.winfo_exists():
                return
        act, held, note, (mdx, mdy) = self._brain_activity()
        # 按键 down/up 事件(边缘检测)
        labels = {k["label"]: k for k in held}
        for k in held:
            if k["label"] not in self.held_prev:
                self.mirror.key_down(k)
        for lab in self.held_prev - set(labels):
            self.mirror.key_up({"label": lab})
        self.held_prev = set(labels)
        self.total_events = self.frames
        self.spark.append(float(np.mean(act)))
        self.frames += 1

        s = self.scale
        img, d, m, _ = self._frame_pil()
        # 布局行
        brain_box = (m, int(46 * s), m + int(258 * s), int(322 * s))
        arr = np.asarray(img).copy()
        if self.show.get("brain", True):
            self._brain(arr, act, brain_box)
            img = self.Image.fromarray(arr)
            d = ImageDraw.Draw(img)
        else:
            d.rectangle(brain_box, fill=PANEL2, outline=BORDER)
            d.text(((brain_box[0] + brain_box[2]) / 2,
                    (brain_box[1] + brain_box[3]) / 2), "脑图已隐藏",
                   font=self.f_sm, fill=DIM_C, anchor="mm")
        # 右列:果蝇 + 神经元统计 + 当前动作
        rx0 = m + int(262 * s)
        self._fly(d, rx0 + int(56 * s), int(100 * s), int(64 * s),
                  0 if not mdx else (1 if mdx > 0 else -1))
        d.text((rx0 + int(6 * s), int(158 * s)), "166,700 个神经元",
               font=self.f_lab, fill=DIM_C)
        d.text((rx0 + int(6 * s), int(172 * s)), "25.58M 个突触",
               font=self.f_lab, fill=DIM_C)
        ol_idx = np.concatenate([self.g_idx[g] for g in
                                 ("ol_sensory", "ol_intrinsic", "visual_projection")
                                 if g in self.g_idx])
        d.text((rx0 + int(6 * s), int(196 * s)),
               f"视叶活动 {float(np.mean(act[ol_idx]))*100:.0f}%",
               font=self.f_lab, fill=DIM_C)
        if held:
            k = held[0]
            d.text((rx0 + int(6 * s), int(216 * s)),
                   f"当前: {k['label']}" +
                   (f" +{len(held)-1}" if len(held) > 1 else ""),
                   font=self.f_sm, fill=ACCENT)
            d.text((rx0 + int(6 * s), int(232 * s)), k["desc"],
                   font=self.f_sm, fill=TITLE_C)
        else:
            d.text((rx0 + int(6 * s), int(216 * s)), "当前: 待机",
                   font=self.f_sm, fill=DIM_C)
        # 全宽键帽行(视频同款,按下的点亮)
        self._keycaps(d, m + int(4 * s), int(330 * s), set(labels),
                      self.W - 2 * m - int(8 * s))
        d.text((m + int(6 * s), int(366 * s)),
               f"本次运行 {self.frames} 帧 · 用时 {time.perf_counter()-self.t_start:.0f}s",
               font=self.f_sm, fill=TITLE_C)
        if self.show.get("stats", True):
            self._stats_grid(d, m + int(6 * s), int(390 * s), act, held, note)
        # 控制增量(鼠标)
        d.text((m + int(6 * s), int(468 * s)),
               f"鼠标增量 Δx {self.mirror.win_dx:+d} · Δy {self.mirror.win_dy:+d} px/s"
               f"   事件 {self.mirror.rate:.1f}/s",
               font=self.f_sm, fill=CYAN)
        # 波形
        if self.show.get("spark", True):
            self._spark(d, m, int(492 * s), self.W - m, int(530 * s))
        # 控制台
        if self.show.get("console", True):
            self._console(d, m, int(536 * s), self.W - m, self.H - m - int(22 * s))
        if self.adjust:
            hint = "调整模式 · 拖动移动 · 滚轮缩放 · Ctrl+Alt+D 完成并恢复穿透"
        elif time.perf_counter() - self.t_start < 10:
            hint = ("Ctrl+Alt+D 调整位置 · Ctrl+Alt+H 隐藏 · "
                    "Ctrl+Alt+Q 退出 · 数据为输入映射")
        else:
            hint = note
        d.text((self.W / 2, self.H - m - int(11 * s)), hint,
               font=self.f_foot, fill=(ACCENT if self.adjust else DIM_C),
               anchor="mm")
        ph = self.ImageTk.PhotoImage(img)
        self._photo = ph
        self.label.config(image=ph)

    def run(self):
        print(f"fly_overlay pid={os.getpid()}", flush=True)
        print("热键: Ctrl+Alt+D 调整(拖动/缩放) | 方向键 移动 | C 穿透 | "
              "X 捕获排除 | H 隐藏 | Q 退出", flush=True)
        self.root.mainloop()
        self.hotkeys.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=int, default=-1)
    ap.add_argument("--y", type=int, default=100)
    ap.add_argument("--scale", type=float, default=0.0, help="覆盖配置的缩放")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--state", default="", help="ann_dashboard 状态文件")
    ap.add_argument("--reset", action="store_true", help="忽略并重置配置")
    args = ap.parse_args()
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            log = CONFIG_PATH.parent / "overlay_crash.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        raise
