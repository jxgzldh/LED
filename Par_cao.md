---
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
---
### CIE1931CIE1960
CIE1931色品坐标转化为CIE1960色品坐标的式子如下：
$$u=\frac {4x}{-2x+12y+3},v=\frac {6y}{-2x+12y+3}$$
若写成python函数，如下：
```python
def xy_to_uv(x, y):
    """
        将CIE1931色品坐标(x,y)转换为CIE1960色品坐标(u,v)
    """
    u = 4 * x / (-2 * x + 12 * y + 3)
    v = 6 * y / (-2 * x + 12 * y + 3)
    return u, v
```
### CIE1931转CIE1976
CIE1931色品坐标转化为CIE1976色品坐标的式子如下：
$$u'=\frac {4x}{-2x+12y+3},v'=\frac {9y}{-2x+12y+3}$$
若写成python函数如下：
```python
def xy_to_udvd(x, y):
    """
	将CIE1931色品坐标(x,y)转换为CIE1976色品坐标(u',v')
    """
    ud = 4 * x / (-2 * x + 12 * y + 3)
    vd = 9 * y / (-2 * x + 12 * y + 3)
    return ud, vd
```
### 由色温求黑体曲线上的坐标
  #### 方法一：近似公式(CIE1960UCS)
  该方法精度不高，误差大，不建议使用。
  计算 $d=\frac{10^8}{T}$
$若 T \leq 7000 \, \text{K} ：$
$$u = \frac{0.860117757 + 1.54118254 \times 10^{-4}d + 1.28641212 \times 10^{-7}d^2}{1 + 8.42420235 \times 10^{-4}d + 7.08145163 \times 10^{-7}d^2}$$
$$v = \frac{0.317398726 + 4.22806245 \times 10^{-5}d + 4.20481691 \times 10^{-8}d^2}{1 - 2.89741816 \times 10^{-5}d + 1.61456053 \times 10^{-7}d^2}$$
$将 uv 转换为 xy(可选)：$
$$x = \frac{3u}{2u - 8v + 4}, \quad y = \frac{2v}{2u - 8v + 4}$$
若写成python函数如下：
```python
import numpy as np

def cct_to_xy(T):
    """
    通过CIE 1960 UCS近似公式，将色温（K）转换为黑体轨迹的xy色坐标。
    适用条件：T <= 7000K（更高温度需更复杂模型）
    """
    d = 1e8 / T
    
    # 计算CIE 1960 UCS坐标 (u, v)
    numerator_u = 0.860117757 + 1.54118254e-4 * d + 1.28641212e-7 * d**2
    denominator_u = 1 + 8.42420235e-4 * d + 7.08145163e-7 * d**2
    u = numerator_u / denominator_u
    
    numerator_v = 0.317398726 + 4.22806245e-5 * d + 4.20481691e-8 * d**2
    denominator_v = 1 - 2.89741816e-5 * d + 1.61456053e-7 * d**2
    v = numerator_v / denominator_v
    
    # 将(u, v)转换为(x, y)
    x = (3 * u) / (2 * u - 8 * v + 4)
    y = (2 * v) / (2 * u - 8 * v + 4)
    return x, y

# 示例：计算6500K的色坐标
x, y = cct_to_xy(6500)
print(f"T=6500K的黑体色坐标 (x, y) = ({x:.4f}, {y:.4f})")
```
#### 方法二:普朗克公式积分法
该方法精度高，但计算量大。
- **普朗克辐射公式**：黑体辐射的光谱分布辐射出射度形式为：
  
$$M(\lambda, T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{hc/(\lambda k T)} - 1}$$

描述单位波长间隔（m）内的辐射出射度
(单位： $\text{W} \cdot \text{m}^{-2} \cdot \text{m}^{-1}$)
  其中：
  - $\lambda$ 为波长
  - $T$ 为色温
  - $h = 6.62607015 \times 10^{-34} J·s$为普朗克常数
  - $c = 299792458\text{m/s}$为真空中的光速
  - $k_B = 1.380649\times10^{-23}\, \text{J/K}$为玻尔兹曼常数
  - **适用场景**：实验测量中常用，因为光谱仪通常以波长标定（如可见光、红外波段）。
- **色坐标计算**：需对 $M(\lambda, T)$在可见光波段（380–780nm）积分，结合CIE 1931色匹配函数 $\overline{x}(\lambda), \overline{y}(\lambda), \overline{z}(\lambda)，计算三刺激值X, Y, Z $。
- 积分计算三刺激值 \( X, Y, Z \)：
  
$$
       X = \int_{380}^{780} M(\lambda, T) \cdot \overline{x}(\lambda) \, d\lambda
$$

$$
       Y = \int_{380}^{780} M(\lambda, T) \cdot \overline{y}(\lambda) \, d\lambda
$$

$$
       Z = \int_{380}^{780} M(\lambda, T) \cdot \overline{z}(\lambda) \, d\lambda
$$

但是一般使用上述积分公式的求和形式近似计算三刺激值。

$$
       X = \quad \sum_{i=380}^{780} M(\lambda_i, T) \cdot \overline{x}(\lambda_i) \, \Delta\lambda
$$

$$
       Y = \quad \sum_{i=380}^{780} M(\lambda_i, T) \cdot \overline{y}(\lambda_i) \, \Delta\lambda
$$

$$
       Z = \quad \sum_{i=380}^{780} M(\lambda_i, T) \cdot \overline{z}(\lambda_i) \, \Delta\lambda
$$

其中 $\Delta\lambda$ 可以是1nm，5nm，但计算时单位是m
- 转换为 \( xy \) 坐标：
- 
$$
       x = \frac{X}{X+Y+Z}, \quad y = \frac{Y}{X+Y+Z}
$$

**波长范围选择**：
- **常规范围**：380–780nm（可见光），步长1nm或5nm（CIE标准数据间隔）。
- **低温扩展**：对低温（如2700K），黑体辐射峰值在红外区（约1070nm），需扩展波长至2500nm（但色匹配函数在780nm外为0）。
- 写成python如下：
```python
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd

#trapezoid()函数是scipy模块里积分方法里的梯形法则来计算数值积分

data_2d = pd.read_excel('/storage/emulated/0/Download/py_game/ldh/2d_1nm.xls', header=None)
x_bar = data_2d.iloc[:, 1].values.flatten()  # 读取B列数据并转化为数组
y_bar = data_2d.iloc[:, 2].values.flatten()  # C列
z_bar = data_2d.iloc[:, 3].values.flatten()  # D列

# 波长范围380-780nm，间隔1nm
wl = np.arange(380, 781, 1)
#辐射出射度计算函数
def planck_spectrum(lam_nm, T):
    lam = lam_nm * 1e-9  # 转换为米
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    #下列条件是避免分母极大时，直接返回0，以免表达式趋于0出现溢出错误
    term = h * c / (lam * k * T)
    if term > 700:
        return 0.0
    #下列判断是避免非法输入
    denominator = np.exp(term) - 1
    if denominator <= 0:
        return 0.0
    return (2 * h * c**2) / (lam**5 * denominator)
#CIE1931色坐标计算
def cct_to_xy(T):
    B = np.array([planck_spectrum(lam, T) for lam in wl])
    X = trapezoid(B * x_bar, wl)
    Y = trapezoid(B * y_bar, wl)
    Z = trapezoid(B * z_bar, wl)
    
    sum_XYZ = X + Y + Z
    if sum_XYZ < 1e-10:
        return (0.0, 0.0)
    x = X / sum_XYZ
    y = Y / sum_XYZ
    return (x, y)

# 输入色温并计算
cct = int(input("输入色温："))
x, y = cct_to_xy(cct)
print(f"T={cct}K的色坐标 (x, y) = ({x:.10f}, {y:.10f})")
```
### 由色坐标求相关色温
1. **将色坐标转换为均匀色空间（如CIE 1960 UCS）：**
   - 从 \( xy \) 转换为 \( uv \)：
     
    $u = \frac{4x}{-2x + 12y + 3}, \quad v = \frac{6y}{-2x + 12y + 3}$
   
2. **数值迭代法（牛顿法）求解CCT：**
   - 目标：找到使黑体轨迹上点 \( (u(T), v(T)) \) 到目标点 $u_{\text{target}},v_{\text{target}}$ 距离最小的 \( T \)。
   - 构造方程（正交条件）：
   
     $(u(T) - u_{\text{target}}) \cdot \frac{du}{dT} + (v(T) - v_{\text{target}}) \cdot \frac{dv}{dT} = 0$

   - 通过牛顿迭代更新 $T$ ：
     
$$T_{n+1} = T_n - \frac{f(T_n)}{f'(T_n)}$$

其中 $f(T)$ 为上述正交条件方程。
1. **近似公式（McCamy公式，适用于3000K–10000K）：**
从 $xy$ 直接计算：

$$
n = \frac{x - 0.3320}{0.1858 - y}, \quad CCT = 449n^3 + 3525n^2 + 6823.3n + 5520.33
$$

写成python函数如下：
```python
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd
from scipy.optimize import minimize
import os
import pickle

# 加载色匹配函数数据
data_2d = pd.read_excel('/storage/emulated/0/Download/py_game/ldh/2d_1nm.xls', header=None)
x_bar = data_2d.iloc[:, 1].values.flatten()
y_bar = data_2d.iloc[:, 2].values.flatten()
z_bar = data_2d.iloc[:, 3].values.flatten()

wl = np.arange(380, 781, 1)  # 380-780nm，1nm步长

def planck_spectrum(lam_nm_array, T):
    lam = lam_nm_array * 1e-9
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    
    term = h * c / (lam * k * T)
    term = np.where(term > 700, np.inf, term)
    denominator = np.exp(term) - 1
    denominator = np.where(denominator <= 0, np.inf, denominator)
    B = (2 * h * c**2) / (lam**5 * denominator)
    B = np.where(term > 700, 0.0, B)
    return B

def cct_to_uv(T):
    B = planck_spectrum(wl, T)
    X = trapezoid(B * x_bar, wl)
    Y = trapezoid(B * y_bar, wl)
    Z = trapezoid(B * z_bar, wl)
    
    sum_XYZ = X + Y + Z
    if sum_XYZ < 1e-10:
        return (0.0, 0.0)
    x = X / sum_XYZ
    y = Y / sum_XYZ
    u = (4 * x) / (-2 * x + 12 * y + 3)
    v = (6 * y) / (-2 * x + 12 * y + 3)
    return (u, v)

def initial_guess_cct(x, y):
    n = (x - 0.3320) / (0.1858 - y)
    return 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

def uv_distance(T_array, uv_target):
    T = T_array[0]
    u, v = cct_to_uv(T)
    return np.sqrt((u - uv_target[0])**2 + (v - uv_target[1])**2)

# 预加载LUT
LUT_FILE = 'blackbody_lut.pkl'
if os.path.exists(LUT_FILE):
    with open(LUT_FILE, 'rb') as f:
        T_range, uv_lut = pickle.load(f)
else:
    T_range = np.arange(500, 15001, 1)  # 调整步长以平衡速度与精度
    uv_lut = np.array([cct_to_uv(T) for T in T_range])
    with open(LUT_FILE, 'wb') as f:
        pickle.dump((T_range, uv_lut), f)

def xy_to_cct(x, y):
    if not (0 <= x <= 0.8 and 0 <= y <= 0.85):
        raise ValueError("色坐标超出合理范围")
    
    u_target = (4 * x) / (-2 * x + 12 * y + 3)
    v_target = (6 * y) / (-2 * x + 12 * y + 3)
    
    T_approx = initial_guess_cct(x, y)
    nearest_idx = np.argmin(np.abs(T_range - T_approx))
    T_initial = T_range[nearest_idx]
    
    result = minimize(uv_distance, x0=np.array([T_initial]), 
                     args=((u_target, v_target),), method='Nelder-Mead', tol=1e-4)
    
    if not result.success:
        return np.nan, np.nan
    
    cct = result.x[0]
    u_cct, v_cct = cct_to_uv(cct)
    duv = np.sqrt((u_target - u_cct)**2 + (v_target - v_cct)**2) * np.sign(v_target - v_cct)
    return cct, duv
def zn_to_en(text):
    # 替换中文逗号为英文逗号
    return text.replace("，", ",")

# 测试
input_str = input("请输入坐标（格式：x,y)：")
x, y = map(float, zn_to_en(input_str).split(','))  # 不管输入中文逗号，还是英文逗号，分割时都为英文字符串
cct, duv = xy_to_cct(x, y)
print(f"CCT = {cct:.4f}K, Duv = {duv:.4f}")
```

### 由(Tc,duv)转换成(x,y)
要将相关色温（Tc）和色温差（Duv）转换为CIE 1931色坐标（x, y），需遵循以下步骤：

**步骤1：确定黑体轨迹上的基础点（Tc对应的(u₀, v₀)）**
- **计算黑体在Tc下的CIE 1931色坐标(x₀, y₀)**  
   使用普朗克公式计算黑体在温度Tc下的光谱辐射分布，积分得到三刺激值XYZ，再转换为CIE 1931色坐标：
  
   $x₀ = \frac{X}{X+Y+Z}, \quad y₀ = \frac{Y}{X+Y+Z}$

- **转换为CIE 1960 UCS坐标(u₀, v₀)**  
   CIE 1960 UCS均匀色空间坐标的转换公式为：
  
   $u₀ = \frac{4x₀}{-2x₀ + 12y₀ + 3}, \quad v₀ = \frac{6y₀}{-2x₀ + 12y₀ + 3}$

**步骤2：计算黑体轨迹在(u₀, v₀)处的法线方向**
+ **获取黑体轨迹的切线方向**  
   对黑体轨迹参数化（例如以色温T为参数），计算其导数得到切线方向。  
1.**数值方法**：选取邻近色温点（如T ± ΔT），计算对应(u₁, v₁)和(u₂, v₂)，则切线方向近似为：

$$
\text{切线向量} = (u₂ - u₁, v₂ - v₁)
$$
	
2. **法线方向**：切线向量的垂直方向（交换分量并取反），归一化为单位向量：
  
$$
\text{法线单位向量} = \frac{(- (v₂ - v₁), u₂ - u₁)}{\sqrt{(v₂ - v₁)^2 + (u₂ - u₁)^2}}
$$

 **步骤3：沿法线方向应用Duv偏移**
-  **计算偏移后的(u', v')**  
   根据Duv的正负号沿法线方向移动：
   
   $u' = u₀ + \text{Duv} \cdot \text{法线单位向量的u分量}$
   
   $v' = v₀ + \text{Duv} \cdot \text{法线单位向量的v分量}$
   
   **注意**：Duv的正负约定需与标准一致（通常正Duv表示色度点位于黑体轨迹外侧）。

**步骤4：将(u', v')转换为CIE 1931色坐标(x, y)**
- **逆转换公式**  
   从CIE 1960 UCS回到CIE 1931的转换公式为：
  
   $x = \frac{3u'}{2u' - 8v' + 4}, \quad y = \frac{2v'}{2u' - 8v' + 4}$

转化为python程序
```python
import numpy as np
from scipy.integrate import simpson  # 改用Simpson积分提高精度
import pandas as pd

# 加载CIE 1931标准观察者色匹配函数数据
data_2d = pd.read_excel('/storage/emulated/0/Download/py_game/ldh/2d_1nm.xls', header=None)
x_bar = data_2d.iloc[:, 1].values.flatten()
y_bar = data_2d.iloc[:, 2].values.flatten()
z_bar = data_2d.iloc[:, 3].values.flatten()

wl = np.arange(380, 781, 1)  # 380-780nm，步长1nm

def planck_spectrum(lam_nm_array, T):
    """ 向量化计算黑体辐射（数值稳定版） """
    lam = lam_nm_array * 1e-9
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    
    term = h * c / (lam * k * T)
    term = np.where(term > 700, np.inf, term)
    denominator = np.exp(term) - 1
    denominator = np.where(denominator <= 0, np.inf, denominator)
    B = (2 * h * c**2) / (lam**5 * denominator)
    B = np.where(term > 700, 0.0, B)
    return B

def cct_to_uv(T):
    """ 计算色温对应的CIE 1960 UCS坐标（使用Simpson积分） """
    B = planck_spectrum(wl, T)
    X = simpson(B * x_bar, wl)
    Y = simpson(B * y_bar, wl)
    Z = simpson(B * z_bar, wl)
    
    sum_XYZ = X + Y + Z
    if sum_XYZ < 1e-10:
        return (0.0, 0.0)
    x = X / sum_XYZ
    y = Y / sum_XYZ
    u = (4 * x) / (-2 * x + 12 * y + 3)
    v = (6 * y) / (-2 * x + 12 * y + 3)
    return (u, v)

def calculate_normal_vector(Tc, delta_T=0.1):
    """
    计算法线方向（中心差分法，delta_T=0.1K）
    返回单位向量，方向为黑体轨迹外侧（正Duv方向）
    """
    T_prev = Tc - delta_T
    T_next = Tc + delta_T
    u_prev, v_prev = cct_to_uv(T_prev)
    u_next, v_next = cct_to_uv(T_next)
    
    # 切线方向（中心差分）
    tangent_u = u_next - u_prev
    tangent_v = v_next - v_prev
    
    # 法线方向（顺时针旋转90度，指向外侧）
    normal_u = tangent_v
    normal_v = -tangent_u
    
    # 单位化
    norm = np.sqrt(normal_u**2 + normal_v**2)
    if norm < 1e-10:
        return (0.0, 0.0)
    return (normal_u / norm, normal_v / norm)

def tc_duv_to_xy(Tc, Duv):
    """ 高精度 (Tc, Duv) 转 (x, y) """
    u0, v0 = cct_to_uv(Tc)
    n_u, n_v = calculate_normal_vector(Tc)
    
    # 沿法线方向应用Duv偏移（正Duv向外侧）
    u_prime = u0 + Duv * n_u
    v_prime = v0 + Duv * n_v
    
    # 转换回CIE 1931
    denominator = 2 * u_prime - 8 * v_prime + 4
    if abs(denominator) < 1e-10:
        return (0.0, 0.0)
    x = (3 * u_prime) / denominator
    y = (2 * v_prime) / denominator
    return (x, y)

def zn_to_en(text):
    return text.replace("，", ",")

# 用户交互部分
while True:
    input_str = input("请输入坐标（格式：Tc,duv）：")
    try:
        Tc, Duv = map(float, zn_to_en(input_str).split(','))
        x, y = tc_duv_to_xy(Tc, Duv)
        print(f"(x, y) = ({x:.4f}, {y:.4f})")
        
    except ValueError:
        print("输入格式错误，请按格式输入（例如：2732,-0.0009）")
    
    choice = input('是否继续计算？（y/n): ')
    if choice.lower() != 'y':
        break
```
### 由(x,y)转换为(主波长或补色波长,色纯度)
##### 1 **点是否在三角形内**  
设三角形顶点为白点 $(x_w, y_w)$ 、380nm 点 $(x_{380}, y_{380})$ 、780nm 点 $(x_{780}, y_{780})$ 。  
用叉积法判断点 $(x, y)$ 是否在三角形内部：  

$$
\begin{cases}
\text{sign}_1 = (x_{380} - x_w)(y - y_w) - (y_{380} - y_w)(x - x_w), \\
\text{sign}_2 = (x_{780} - x_{380})(y - y_{380}) - (y_{780} - y_{380})(x - x_{380}), \\
\text{sign}_3 = (x_w - x_{780})(y - y_{780}) - (y_w - y_{780})(x - x_{780})
\end{cases}
$$ 

若 $\text{sign}_1, \text{sign}_2, \text{sign}_3$ 同号，则点在三角形内，对应补色波长；否则对应主波长。

##### 2 **主波长计算**  
- **方向向量**：$\vec{d} = (x - x_w, y - y_w)$。  
- **投影参数**：对光谱轨迹点 $(x_\lambda, y_\lambda)$，计算投影参数：
  
  $$
  t = \frac{(x_\lambda - x_w)(x - x_w) + (y_\lambda - y_w)(y - y_w)}{\|\vec{d}\|^2}, \quad \text{保留 } t \geq 0.
  $$
   
- **投影点**：$(x_p, y_p) = (x_w + t(x - x_w), y_w + t(y - y_w))$。  
- **最小距离**：找到使 $\Delta^2 = (x_p - x_\lambda)^2 + (y_p - y_\lambda)^2$ 最小的 $\lambda$。  
- **纯度**：
    
  $$
  p = \min\left( \frac{\sqrt{(x - x_w)^2 + (y - y_w)^2}}{\sqrt{(x_\lambda - x_w)^2 + (y_\lambda - y_w)^2}}, \ 1 \right)
  $$ 

##### 3 **补色波长计算**  
- **反向投影**：方向向量取 $\vec{d}' = (x_w - x, y_w - y)$，其余步骤与主波长相同。  
- **波长符号**：$\lambda_{\text{comp}} = -\lambda$。
- **纯度公式**：
    
  $p = \min\left( \frac{\|\overrightarrow{wp}\|}{\|\overrightarrow{w\lambda}\|}, \ 1 \right),$

  其中 $\overrightarrow{w\lambda}$ 是白点到光谱轨迹点的向量。  
- **数值稳定性**：若 $\|\overrightarrow{w\lambda}\| < \epsilon$（如 $\epsilon = 10^{-10}$），则设 $p = 0$。  
由python实现如下
```python
import numpy as np
import pandas as pd

# 加载CIE 1931光谱轨迹数据（0.1nm间隔）
data_wld = pd.read_excel('/storage/emulated/0/Download/py_game/ldh/wld_xy.xls', header=None)
wavelengths = data_wld.iloc[:, 0].values.astype(float)
x_spectral = data_wld.iloc[:, 1].values.astype(float)
y_spectral = data_wld.iloc[:, 2].values.astype(float)

# 定义白点和紫线端点
x_white, y_white = 0.3333, 0.3333
x_380, y_380 = x_spectral[0], y_spectral[0]
x_780, y_780 = x_spectral[-1], y_spectral[-1]

# 预计算光谱轨迹与白点的相对坐标（向量化）
dx_spectral = x_spectral - x_white
dy_spectral = y_spectral - y_white

def is_inside_triangle(x, y):
    """判断点是否在三角形内部（白点、380nm、780nm）"""
    v0 = np.array([x_white, y_white, 0])
    v1 = np.array([x_380, y_380, 0])
    v2 = np.array([x_780, y_780, 0])
    p = np.array([x, y, 0])
    
    sign1 = np.cross(v1 - v0, p - v0)[2]
    sign2 = np.cross(v2 - v1, p - v1)[2]
    sign3 = np.cross(v0 - v2, p - v2)[2]
    return (sign1 > 1e-10) & (sign2 > 1e-10) & (sign3 > 1e-10)

def find_complementary_wavelength_fast(x_target, y_target):
    """快速补色波长计算（优化版）"""
    dx = x_white - x_target
    dy = y_white - y_target
    dir_norm_sq = dx**2 + dy**2
    
    if dir_norm_sq < 1e-12:
        return 0.0, 0.0, 0.0
    
    # 使用预计算的dx_spectral和dy_spectral
    t_numerator = dx_spectral * dx + dy_spectral * dy
    t = t_numerator / dir_norm_sq
    
    # 预过滤有效索引，减少后续计算量
    valid_indices = np.where(t >= 0)[0]
    if valid_indices.size == 0:
        return 0.0, 0.0, 0.0
    
    t_valid = t[valid_indices]
    x_proj = x_white + t_valid * dx
    y_proj = y_white + t_valid * dy
    
    # 仅计算有效点的距离平方
    dx_diff = x_proj - x_spectral[valid_indices]
    dy_diff = y_proj - y_spectral[valid_indices]
    distances_sq = dx_diff**2 + dy_diff**2
    
    min_idx_valid = np.argmin(distances_sq)
    min_idx = valid_indices[min_idx_valid]
    
    return -wavelengths[min_idx], x_spectral[min_idx], y_spectral[min_idx]

def find_dominant_wavelength_ultrafast(x, y):
    """主函数（优化版）"""
    if is_inside_triangle(x, y):
        comp_wl, xc, yc = find_complementary_wavelength_fast(x, y)
        if comp_wl == 0.0:
            return (0.0, 0.0)
        
        d_sample = np.hypot(x - x_white, y - y_white)
        d_lambda = np.hypot(xc - x_white, yc - y_white)
        purity = min(d_sample / d_lambda, 1.0) if d_lambda > 1e-10 else 0.0
        return (comp_wl, purity)
    else:
        dx = x - x_white
        dy = y - y_white
        dir_norm_sq = dx**2 + dy**2
        
        if dir_norm_sq < 1e-12:
            return (0.0, 0.0)
        
        t_numerator = dx_spectral * dx + dy_spectral * dy
        t = t_numerator / dir_norm_sq
        
        valid_indices = np.where(t >= 0)[0]
        if valid_indices.size == 0:
            return (0.0, 0.0)
        
        t_valid = t[valid_indices]
        x_proj = x_white + t_valid * dx
        y_proj = y_white + t_valid * dy
        
        dx_diff = x_proj - x_spectral[valid_indices]
        dy_diff = y_proj - y_spectral[valid_indices]
        distances_sq = dx_diff**2 + dy_diff**2
        
        min_idx_valid = np.argmin(distances_sq)
        min_idx = valid_indices[min_idx_valid]
        
        dominant_wl = wavelengths[min_idx]
        d_sample = np.sqrt(dir_norm_sq)
        d_lambda = np.hypot(x_spectral[min_idx] - x_white, y_spectral[min_idx] - y_white)
        purity = min(d_sample / d_lambda, 1.0) if d_lambda > 1e-10 else 0.0
        
        return (dominant_wl, purity)

def zn_to_en(text):
    # 替换中文逗号为英文逗号
    return text.replace("，", ",")
while True:
    input_str = input("请输入坐标（格式：x,y）：")
    try:
        x, y = map(float, zn_to_en(input_str).split(','))
        wl, purity = find_dominant_wavelength_ultrafast(x, y)
        print(f"(主波长或补色波长, 色纯度) = ({wl:.1f}, {purity:.4f})")
        
    except ValueError:
        print("输入格式错误，请按格式输入（例如：0.4578,0.4101）")
    
    choice = input('是否继续计算？（y/n): ')
    if choice.lower() != 'y':
        break
```
### 由(主波长或补色波长,色纯度)转化为(x,y)
- **光谱轨迹点**：设白点为 $(x_w, y_w)$，波长 $\lambda$ 对应的光谱轨迹点为 $(x_\lambda, y_\lambda)$。  
- **方向与缩放**：  
  - **主波长（ $ \lambda \geq 0$ ）**：  
    从白点沿方向 $\overrightarrow{w\lambda} = (x_\lambda - x_w, y_\lambda - y_w)$ 按比例 $p$ 移动：  

    $x = x_w + p(x_\lambda - x_w), \quad y = y_w + p(y_\lambda - y_w)$
    
  - **补色波长（ $ \lambda < 0$ ）**：  
    取绝对值波长 $|\lambda|$ 对应的光谱点 $(x_{|\lambda|}, y_{|\lambda|})$，从白点沿反方向 $\overrightarrow{\lambda w} = (x_w - x_{|\lambda|}, y_w - y_{|\lambda|})$ 按比例 $p$ 移动：  

    $x = x_w - p(x_{|\lambda|} - x_w), \quad y = y_w - p(y_{|\lambda|} - y_w)$
    
- **约束条件**：  

  $x = \text{clip}(x, 0, 1), \quad y = \text{clip}(y, 0, 1)$
  
由python实现如下：
```python
import numpy as np
import pandas as pd

# 加载光谱轨迹数据
data_wld = pd.read_excel('wld_xy.xls', header=None)
wavelengths = data_wld.iloc[:, 0].values.astype(float)
x_spectral = data_wld.iloc[:, 1].values.astype(float)
y_spectral = data_wld.iloc[:, 2].values.astype(float)

# 白点坐标
x_white, y_white = 0.3333, 0.3333

def wavelength_to_xy(wl, purity):
    """将波长和纯度转换为色坐标 (x, y)"""
    # 1. 查找波长对应的光谱轨迹点
    if wl < 0:
        # 补色波长：取绝对值并反向延长
        wl_abs = abs(wl)
        idx = np.argmin(np.abs(wavelengths - wl_abs))
        sign = -1
    else:
        # 主波长：直接查找
        idx = np.argmin(np.abs(wavelengths - wl))
        sign = 1
    
    x_lambda = x_spectral[idx]
    y_lambda = y_spectral[idx]
    
    # 2. 计算色坐标
    x = x_white + sign * purity * (x_lambda - x_white)
    y = y_white + sign * purity * (y_lambda - y_white)
    
    # 限制坐标在 [0, 1] 范围内（物理意义约束）
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return x, y
def zn_to_en(text):
    # 替换中文逗号为英文逗号
    return text.replace("，", ",")
# 用户交互
while True:
    input_str = input("请输入坐标（格式：主波长或补色波长,色纯度）：")
    try:
        wl, purity = map(float, zn_to_en(input_str).split(','))
        x, y = wavelength_to_xy(wl, purity)
        print(f"(x, y) = ({x:.4f}, {y:.4f})")
        
    except ValueError:
        print("输入格式错误，请按格式输入（例如：584.3,0.5910）")
    
    choice = input('是否继续计算？（y/n): ')
    if choice.lower() != 'y':
        break
```
