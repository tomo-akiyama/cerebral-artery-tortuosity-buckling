# Comments from Jupyter Notebook: 251129 sin+trend fitting データを1被験者ずつ抽出のコピー.ipynb

## Cell 8

# ユーティリティ

## Cell 10

# まず被験者内の全体の統計量

## Cell 16

# Segment idを指定

## Cell 17

# Step 1（マクロな説明）

### outcome: 
### κ_orig_RMS
### D, L, TI(=L/D) & 余剰長ρ(=L/D−1)
### linearity, planarity1, scattering, planarity2      

## Cell 21

Power_DCTグラフについて.  
sin+trend で当てたあと、どの周波数帯のエネルギーがどれくらい減ったかを大まかに知りたい.  
k_hat 付近（推定した波数）にあったパワーは、residual でちゃんと減っているか？   
残差のパワーは、orig と比べて全体的に何桁くらい小さくなっているか？   
まだ大きなピークが残っている周波数はあるか？（＝高調波や別モード）.  

## Cell 26

# MCA segment: M1, M2, M3, M4
上のpyvistaをみてここは手動で指定する必要あり.  
=> M2〜M3 にまたがる幹などを考えると、M1 (sylvian cistern, sphenoidal) / M2–3 (sylvian cistern, operculoinsular) / M4 の3区分.

※代表点（centroid）ベースで決める!!   
Segments that spanned more than one compartment were assigned to the compartment containing the centroid of the segment, defined as the mean of all centerline coordinates.

## Cell 30

※ https://ja.wikipedia.org/wiki/細長比#:~:text=細長比（ほそながひ、さ,値が決定される%E3%80%82&text=なお、有効長さ係数,を用いることもある%E3%80%82.  

構造力学では、柱や梁のスレンダー比はL/r_g (ただしここでのr_gは断面回転半径"radius of gyration").  
r_g = sqrt(I/A), Iは断面二次モーメント、Aは断面積.  
円柱の断面2次モーメント I = pi*r^4 / 4, A = pi*r^2 つまり I/A = r^2 / 4 (ここでのrは半径).  
したがって断面回転半径r_g = r/2 = D/4. 
よってスレンダー比 L/r_g = 4 * L/D

※ https://fiveable.me/statics-strength-materials/unit-15/eulers-formula-limitations/study-guide/6kD22wHNwNW1MuSc?utm_source=chatgpt.com.  
によるとL/r > 100：長柱（Euler buckling が支配的）. 
つまり L/D > 25 が理想

※ https://www.jpe-innovations.com/precision-point/beam-theory-bending/?utm_source=chatgpt.com.  
によると L/h(ここでのhは直径) > 10 -> slender beamとみなしてOK  

## Cell 33

### 選んだセグメントが左右どちらか自動判定させる

## Cell 37

# 平面性の情報

## Cell 39

MCAセグメントで1D buckling を“2D 平面内の棒”としてモデル化
=>
必要条件として $\lambda_3 \ll \lambda_1,\lambda_2 \quad\text{(＝Sλ がかなり小さい)}$. ※厳密に平面上ならばλ3=0.   
さらに rod-like であってほしいなら $L_\lambda \text{も十分に大きい（λ₁≫λ₂）}$.  
という 2 つをセット

## Cell 40

https://www.researchgate.net/figure/a-Visualization-of-1-linearity-2-planarity-and-3-sphericity-after-43-b_fig4_355202086

linearity (L λ )
planarity (P λ )
sphericity (S λ )
omnivariance (O λ )
anisotropy (A λ )
change of curvature (C λ ).
と各種の定義の図がある

## Cell 42

Planarity: 
≥ 0.95 ほぼ完全に 1 枚の平面上 Planar segment（平面内走行） -> 1D buckling model 適用可
0.90–0.95 わずかに 3D 的（軽度ねじれ or 脳表の湾曲追従）-> Mostly planar（planar buckling 近似は妥当）

## Cell 43

	planarity = 1 − λ₃/(λ₁ + λ₂ + λ₃)
　→ λ₃ が λ₁, λ₂ に比べて極端に小さい（＝3D方向の分散がほとんどない）とき、planarity ≈ 1。
　→ つまり「ほぼ1枚の平面内に点群が分布している」ことを意味する。

planarity ≈ 1 なら：
	•	血管中心線は局所的に「平面帯上」に収まっている
	•	すなわち、1枚の平面内に射影しても情報をほとんど失わない
→ だから1D bucklingで十分に近似可能、といってよいのでは


Planarity ≈ 1 は「血管中心線が局所的に1枚の薄い曲面（≒脳表）上に沿っている」 
=>平面上で座屈することを保証するわけではないが、1D buckling modelの必要条件（仮定）にはなる
necessary condition（必要条件）, not sufficient condition（十分条件）


PCA における planarity ≈ 1 は、「数学的には1枚の平面で近似できる」という意味であり、
それは実際の幾何構造が“真に平面”であることではなく、
“滑らかな曲面（帯状構造）の局所部分が平面近似可能なほど緩やか”であることを示す
=>The PCA-based planarity was treated as a "necessary geometric condition" for applying the 1D buckling model, ensuring that each vessel segment can be approximated by a single local plane.
PCA planarity ≈ 1（全体平面性）: セグメント全体を 1 枚の平面で近似できる。モデル適用が最もシンプル。
ただし、局所平面性がなりたっても1D buckling modelは適応可能ではある。   


planarity が高い（λ₃ ≪ λ₂）.   = この枝の 3D 形状は、そもそも“どこかの平面”の近くに収まっているのか？」（＝完全3Dランダム蛇行ではないか？）
-> この枝は、少なくとも「ある 2D 平面」の近くに収まっている.  
->「平面 buckling モデルを当てはめる土俵はありそう」.  


## Cell 46

# Step 2（ミクロな buckling 成分）

## Cell 48

# Bucklingの前提としてL >= 10mmとし、< 10mmは除外

## Cell 49

本研究では，解析対象とする周期波長の下限を 10 mm とした。この値は，単なる便宜的な数値ではなく，撮像分解能と血管径スケールの両方を踏まえた保守的な下限として設定したものである。

本データの空間分解能は 0.5 mm/voxel であり，partial volume 効果を考えると，脳表動脈が「血管として」識別可能となるのは少なくとも 3–4 voxel 程度の直径（約 1.5–2 mm）を有する場合に限られると考えられる。そのため，解析対象とする血管半径の下限をおおよそ 0.5 mm（=1 voxel）とみなすのが保守的である。さらに，本研究では既報の buckling モデルおよび M1 径のスケールをもとに，座屈波長が血管半径に比例すると仮定し，
$\lambda_{\text{prior}} = \alpha r,\quad \alpha \approx 18.9$
という関係を用いた。このとき最小半径 r = 0.5 mm を代入すると，事前波長は
$\lambda_{\text{prior}} \approx 9.45\ \text{mm}$
となり，約 1 cm が「物理モデルと空間分解能の両方からみた，実質的な最小スケール」となる。

波長がこれより短い領域では，1 周期あたりに含まれる有効ボクセル数が少なく，中心線抽出誤差やノイズの影響が卓越しやすくなるため，「見かけ上のうねり」が真の座屈モードかどうかを安定して判別することが困難である。以上より，本研究では 10 mm 未満の波長スケールは buckling 波長としては信頼性が低いと判断し，解析対象とする波長／セグメント長の下限を 10 mm に設定した。

⸻

必要なら，Methods ではもう少し短くして：

空間分解能（0.5 mm/voxel）と血管径スケール，および $\lambda \approx 18.9\,r$ という座屈モデルに基づき，半径 0.5 mm の最小血管に対する事前座屈波長は約 9.5 mm と見積もられた．このスケールを下回ると，1 周期あたりのボクセル数が不足しノイズや中心線抽出誤差の影響が大きくなるため，本研究では座屈波長およびセグメント長の下限を一律 10 mmとした．

## Cell 50

なお，端点の不安定性を避けるため，各セグメントの両端約 1 mm をトリムし，その内側（有効長 ≈ 0.8 L）を用いて DCT および sin+trend フィッティングを行った．buckling 波長の主解析には，さらに $L_{\text{trim}}/\lambda_{\text{hat}} \ge 1.5$ などの QC 条件を満たすセグメント（Core 群）のみを用いた．

## Cell 51

# λ ≈ α·rの"α"について
解析では$\lambda \approx \alpha \times r$を使って$\lambda_{prior}$（sin+trend フィットの探索窓のスケール）を作りたい。    
$\alpha$は“正確な物理定数”ではなくスケールのseed。最終の$\hat\lambda$はsin+trend fittingで決め直す。        
=> $\alpha$を“頑健に”・“簡単に”推定したい

## Cell 52

加齢で増えるM1の形はM1はC型なのかS型なのか？   
=>.  
1968Cerebral angiography Krayenbühl.pdf.  
2020Interv Neurol Khatri.pdf.  
S or C型の報告と思っていたが、基本的にC型だけであった.  

===>
#### 251125 alphaの決め方 (251116 データを1被験者ずつ抽出 3の検証の続きの続き secondary修正).ipynb　を参照

## Cell 54

# 結局Savitzky-Golayフィルタいらない?

入力： VMTKで抽出されたcenterline（moving-average smoothing済み）.  
→ これを$C_{orig}(s)$とする.  

Step 1. 大域トレンド用 baseline P_0(s) の作成（1〜2 次多項式）：    
「超低周波な大域トレンド（血管全体の進行方向）」だけを表現する線として使う。   

Step 2. 変位ベクトル$\Delta(s)$の定義：   
VMTK centerline から多項式トレンドを引いて、$\Delta(s) = C_{orig}(s) - P_0(s)$.     

Step 3. Bishop frameの構成：    
フレーム用centerlineはsmoothing済みのVMTK centerlineをそのまま使う。$C_{frame}(s) := C_{orig}(s)$.   
=> $C_{frame}$から接線ベクトル$\mathbf T(s)$を計算。   
=> $\mathbf T(s)$に対してparallel transport（回転最小条件）で法線ベクトル$\mathbf U(s), \mathbf V(s)$を逐次更新し、Bishop frame ($\mathbf T, \mathbf U, \mathbf V$) を得る。    
※これにより、フレームは“脳表に沿って走る VMTK centerline”に沿った、滑らかな局所座標系になる.  
※Bishop frameに求めたい性質は大きく2つ：   
1.ノイズでガタガタしていないこと.  
2.「何に対して横方向を測りたいか」という“物理的基準”に素直であること.  
=> 今回の「物理的基準」は：   
血管は 脳表に沿って走っている.  
uVを「脳表法線方向と整合する成分」として解釈したい.  
=> T(s)は“脳表に沿って走る血管の実際の向き”を追いかけてほしい。= Bishop frameをP_vox_3darr上で作る

Step 4. $\Delta$の接線・法線分解と$u_U, u_V$の計算
各sで$\Delta(s)$を接線＋法線に分解.  
=> 「トレンドを引いた変位$\Delta(s)$を、“脳表に沿うフレーム” 上の横方向2成分に落とした信号$u_U, u_V$を得る。

Step 5. 前処理（detrend & taper）.  
$u_U(s), u_V(s)$に対して.  
軽い線形 detrend（simple_detrend）.  
→ 残っている一次トレンドを除去.  
端部テーパー（taper_window; Hann 等）.  
→ スペクトル漏れを抑える。   
※これにより、後の DCT/FFT と sin+trend フィットが安定する。

Step 6. 周波数（波数）候補kの抽出（seed）.  
前処理した$u_U, u_V$にDCT/FFTをかけてパワースペクトルを計算。       
=> $λ_{prior}$があればその近傍を優先しつつ、ピークの目立つ波数kを複数個 “seed” として選ぶ。   

Step 7. $u_U, u_V$のsin+trend同時フィット.  
各候補波数kについて、同一のkを用いて.   
$u_U(s) \approx a_U \cos(ks) + b_U \sin(ks) + c_{U0} + c_{U1}s (+ c_{U2}s^2)$.  
$u_V(s) \approx a_V \cos(ks) + b_V \sin(ks) + c_{V0} + c_{V1}s (+ c_{V2}s^2)$.  
を最小二乗（OLS or WLS）で同時フィット。   
=> 上式に加えて、「trend のみ」のモデル（cos, sin項なし）もフィットしておき、それぞれの残差から以下を計算：   
RG：trend 単独 vs sin+trend の残差分散の改善率（Residual Gain）.  
R2：決定係数.  
SNR：信号対ノイズ比（定義は実装に準拠）.  
=> 振幅パラメータは.  
$A_U = \sqrt{a_U^2 + b_U^2},\quad$.  
$A_V = \sqrt{a_V^2 + b_V^2}$.  
$A = \sqrt{A_U^2 + A_V^2}$.  
とし、U–V 平面の回転に不変な実効振幅 A を buckling の強さとして採用。

Step 8. 最良波数 k_{\text{best}} と最終パラメータの決定.  
複数の候補kの中から、    
＊RGが最大のものを第一優先.   
＊RGが近い場合はR2の高い方を採用.  
選ばれたkの±20% などの範囲で細かいステップのグリッド探索を行い、$k_{best}$をさらに精密化。    
=> $k_{best}$から$\lambda_{best} = \frac{2\pi}{k_{best}}$を計算。      
=> 最終的な指標として、各セグメントについて.  
＊実効振幅 A.  
＊波長 $\lambda_{best}$.  
＊RG, R2, SNR.     
coverage（セグメント長に対して何波長含まれるか）.  
などを出力。       
※$k_{best}$が探索範囲の端に張り付いた場合はboundary_hitフラグを立て、必要なら後から探索範囲を広げて再評価できるようにしておく。    

Step 9. 全体の伸長率（オプション）.  
bucklingとは別に、血管の全体的な「伸び」を評価したい場合：      
1.	VMTK centerline の弧長：   
$L_{orig} = \int_{s_{\min}}^{s_{\max}} \lVert C'_{\text{orig}}(s)\rVert ds$.   
2.	大域トレンド P_0 の弧長：   
$L_0 = \int_{s_{\min}}^{s_{\max}} \lVert P_0'(s)\rVert ds$.  
3.	相対伸長率：   
$\text{Elongation} = \frac{L_{orig} - L_0}{L_0}$.  
4.	これを 「全体としてどれくらい伸びているか」 の指標として、buckling指標（A, $\lambda_{best}$）と並べて報告する。


## Cell 64

$\boxed{
\text{法平面への射影が 0 } \;\Longleftrightarrow\;
\frac{d\mathbf v}{ds} \text{ は }\mathbf T\text{ 方向成分だけ}
}$

## Cell 66

同定: sin + trend, 2成分同時
\begin{aligned}
u_U(s) &\approx a_U\cos ks + b_U\sin ks + c_U + d_U s (+e_U s^2),\\
u_V(s) &\approx a_V\cos ks + b_V\sin ks + c_V + d_V s (+e_V s^2),
\end{aligned}
(必要時に$s^2$採否)
でkを近傍グリッド検索 ※FFT/DCTヒント+$[0.5, 2.0]\lambda_{prior}$ (中心は priorまたはu_U（または$\sqrt{u_U^2+u_V^2}$）のDCT/FFT ピーク、±20--30\% を粗->細の2段）。

係数から$ A=\sqrt{a_U^2+b_U^2+a_V^2+b_V^2},\lambda_{obs}=2\pi/k$および$R^2$を求める.  
(回転不変の総振幅。楕円性にも対応）.  


方法1) u_UだけをFFT/DCT.  
問題：U 軸の向きは任意。データが U 軸に対して回転していると、信号の一部が V に逃げる -> 見えにくい。   

方法2) $r(s)=\sqrt{u_U^2+u_V^2}$の FFT/DCT.  
問題1：混在。面内の座屈（u_U）と面外の地形（u_V）を足し合わせるので、混ざる。   
問題2：整流バイアス。|\sin ks| のような“絶対値”は、2倍の周波数2kに強いピークを作る（基本波kが弱まり、偶数高調波が強調される）。   

方法3)  複素信号スペクト.  
$c(s)=u_U(s)\;+\;i\,u_V(s)$という複素数列を作り、これにFFT/DCTをかけて振幅スペクトル|C(k)|をみる.  

方法4) パワー和スペクトル.  
$S(k)=\bigl|\mathrm{FFT}(u_U)\bigr|^2\;+\;\bigl|\mathrm{FFT}(u_V)\bigr|^2$

## Cell 67

(1) 短いセグメントにはDCTが相性よい：端点が“つながっていない”短区間でも漏れ（リープ）が出にくく、初期の波長推定に向き。DCT‑II（norm='ortho'）。その後はsin+trendの同時回帰でkを微調整。   
(2) パワー和スペクトルは整流バイアスを作らない：各成分u_U,u_Vに線形変換（FFT/DCT）をかけ、二乗を足すだけなので、|\cos|による倍周波（2k）増強のような“整流”由来の人工ピークが出ない。さらにU–Vの基底回転に対して不変。   

=>

1).	u_U, u_Vそれぞれをデトレンド（平均・一次、必要なら二次まで除去）。   
2).	DCT‑IIを両方にかける（scipy.fft.dct(type=2, norm='ortho')）。   
3).	パワー和$S(m)=|\mathrm{DCT}(u_U)|^2+|\mathrm{DCT}(u_V)|^2$の最大ピークから“初期の”kを得る（後述の回転不変）。   
4).	そのkを初期値に、二成分同時のsin+trend回帰でA,\lambda,R^2を仕上げ推定。
（DCTは端点仮定がFFTと違うため、数％の微調整を回帰で吸収するのが実務的。）.  

※短区間ではFFTでもHannなどの窓掛けを強く推奨されるが、DCTなら窓無しor弱窓でも破綻しにくいのが利点。

## Cell 68

## 実際の手順
1).	前処理（軽いデトレンド）. 
端点の傾きや緩い曲がりがスペクトルへ漏れないよう、u_U,u_V から平均・一次（必要なら二次）トレンドを差し引く。　　
理由: DCT/FFT は直流や低周波に敏感。軽いデトレンドで主ピークが見やすくなる。  
2). DCT によるパワー和スペクトル.  
$U_m=\mathrm{DCT}(u_U),\quad V_m=\mathrm{DCT}(u_V),\quad S_m=|U_m|^2+|V_m|^2 \quad (m=0,1,\dots)$ 　  
の S_m を最大化する \(m^\\) を取り、近似波数 \(k^\\approx \pi m^\/(N\,ds)\) から初期波長 \(\lambda_0=2\pi/k^\\) を得る。　　  
理由: 「パワーの足し算」は 座標回転に不変で、整流（|·|）バイアスで倍周波が偽ピークになる問題を避けられます。短い区間でも DCT は漏れが少なく、初期推定器として優秀。　　

3).	事前帯（任意・推奨）でピーク探索
$\lambda_{\text{prior}}=\alpha r$ を持つなら、
$\lambda\in[0.5,2.0]\times \lambda_{\text{prior}}$（≒±2倍 幅）の現実的な帯域に限定してピークを選ぶ。
理由: 理論スケーリングに整合しない極端なピークの混入を防ぐ。

4).	二成分同時 sin+trend 回帰で仕上げ
\begin{aligned}
u_U(s)&\approx a_U\cos ks+b_U\sin ks+c_U+d_U s(+e_U s^2),\\
u_V(s)&\approx a_V\cos ks+b_V\sin ks+c_V+d_V s(+e_V s^2),
\end{aligned}
とし、同じ波数 k を共有したまま u_U,u_V を同時に最小二乗。
k は DCT 初期値の ±20–30% をグリッド探索して R^2 最大のものを採用。
総振幅（回転不変）
$\displaystyle A=\sqrt{a_U^2+b_U^2+a_V^2+b_V^2}、\lambda=2\pi/k、R^2, n=L/\lambda$を出す。    
理由: DCT の端点仮定差を最後に吸収／微調整でき、U–V の位相差があっても総振幅 A は正しく評価できる。   

## Cell 69

### DCTを選んだ理由
各セグメント内の支配的な空間周期の候補を得るため，弧長パラメータに沿った横変位 u_U(s),u_V(s) に対して type-II 離散コサイン変換（DCT-II）を適用した。DCT-II は，有限区間上の実信号を偶関数として延長したものに対する FFT と数学的に同値であり，有限長・滑らかな信号に対して端点でのスペクトル漏れを抑制できるという利点がある。本研究では中心線は VMTK により平滑化され，さらに両端 1 mm をトリムしているため，DCT-II の仮定する境界条件（端点での勾配が小さい）が妥当と考えられる。なお，DCT スペクトルはあくまで支配波数の候補を与えるために用い，最終的な波長・振幅は空間領域での正弦＋トレンド回帰から推定した。

## Cell 78

DCTにおいて.  
m=1 のエネルギーが高い理由: セグメント全体が「ゆっくり大きく曲がっている」C字（DCTではm=1が半波）.  
m=2 が比較的高い理由：セグメント全長で2回ゆっくり曲がる成分（1周期）、大きなC字の非対称（片側が強い/弱い）などを吸収.  
=> global bednding ()=! local buckling)

## Cell 79

1.	m=1,2 に大きなエネルギーが集中する理由の説明.  
DCT-II m=1 は0.5波の最長波長モードであり、M1 全体の大きなC字カーブ（走行そのもの）がここに現れる。
これは力学的座屈ではなく，解剖学的な大局的走行である。
2.	物理モデルに基づく prior band の導入
座屈理論から $\lambda_{prior} = \alpha r$を定め、その0.5–2倍を「座屈スケールが現れる周波数帯」としてk_bandに定義。    
= この帯域外のエネルギーは「座屈以外（global curvature, ノイズ）」とみなして解析から外す。   
3.	ピークの判定は「バンド内」で行う.  
DCT パワー S[m] のピークは k_band 内に限定して探索。   
m=1,2のような超長波成分はバンド外にあるため，主波長の候補からは除外される。   
4.	エネルギーの “絶対量” ではなく “バンド内での相対優位性” を見る
例えば$S[m_{star}] / E_{band}$や$S[m_{star}]/S_{2nd peak}$を示し、「座屈波長帯の中では m_* が卓越している」ことを定量的に示す。   
5.	補足図として典型例のスペクトルを載せる
まさに今作っているk–power図を補足に載せて、「m=1,2 が global bend を表し，m=4 が座屈スケールでピークになっている」という代表例を示す。

## Cell 80

最大ピークをそのまま採る」では不十分か.  

DCT は有限長・境界条件（偶対称）・非正弦波形の影響で….  
	1.	長波（C 字の大局曲率）の残りが低 k 側に別ピークを作る.  
	2.	基線が十分に消えないと半端な傾き・弦成分が低 k に漏れる.  
	3.	波が完全な正弦でないと倍周波(整数倍の2k, 3k …) にや隣接 k にエネルギー分散が起きる.  
	4.	uU/uV の**楕円性（面内・面外の混合）**でもパワーが割れる.  

…ため、“最大ピーク＝真の基本波”が常に正しいとは限らない.  

## Cell 81

“候補ピークに対して sin+trend（2 成分同時）で当てはめ→適合度で選ぶ”ほうが一貫して頑健なはず.  
近接する複数ピークが出たときは、(1) 当てはめの R²、(2) 3 倍高調波の少なさ、(3) 理論値 k_prior への近さを組にして判定.  


===具体的に===.  
	1.	候補抽出.  
帯域内の局所ピークを最大 3 個ほど抽出（パワー和 S=|U|^2+|V|^2 で上位、閾値例：帯域内最大の 30% 以上）。   
	2.	各候補 k で 2 成分同時回帰（sin+trend）.  
\begin{aligned}
u_U(s)&\approx a_U\cos ks + b_U\sin ks + c_U + d_U s\,(+e_U s^2),\\
u_V(s)&\approx a_V\cos ks + b_V\sin ks + c_V + d_V s\,(+e_V s^2)
\end{aligned}
→ 総合適合度 R^2_{\rm tot} を計算（U と V を縦に連結して 1 本の回帰にするか、二つの SSE を足し合わせる）。      
	3.	3 倍高調波ペナルティ.  
同じ k で |U(3k)|^2+|V(3k)|^2 を |U(k)|^2+|V(k)|^2 で割った比 h（高調波比）を計算。   
→ 基本波がきれいなら h は小さい（例：\le 0.15 を推奨基準）。   
	4.	理論値への近さ（弱い正則化）.  
$\delta=\bigl|\tfrac{k-k_{\rm prior}}{k_{\rm prior}}\bigr|$（近いほど減点が小さい）。   
	5.	スコアで選ぶ（閾値＋タイブレーク）.  
例1：正規化したスコア（しきい値に基づく 0–1 化）.     
J(k)=R^2_{\rm tot} - 0.3\,h - 0.2\,\delta^2   
	•	必須閾値：R^2_{\rm tot}\ge 0.6、周期数 n=L/\lambda(k)\ge 1.5 を満たす候補のみ比較.  
	•	タイブレーク：上位 2 候補の J 差が <0.05 なら k_prior に近い方を採用.  
例2: レキシコグラフィ（段階選抜.  R^2→h_3→|\delta）— 重みの恣意性を避けたいとき.  
1.	ハード条件で足切り：帯域内、R^2\ge 0.6、デトレンド後の残差が許容内….  
	2.	一次基準：R^2 が最大の k を候補に。   
	3.	同点（±0.02以内）なら：h_3 が小さい方。   
	4.	それでも同点なら：|\delta| が小さい方。   
	5.	まだ決まらなければ：より長波（小さい k）を優先（パースモニ：過剰に細かい波は避ける）。   


直感：時系列（ここでは弧長方向）に戻って“本当に正弦で説明できているか”を評価することが、単純なスペクトル最大化より信頼できる。   

## Cell 85

# sin + trend（二成分同時回帰）
### 信号を「正弦波成分（sin, cos）」＋「トレンド成分（平均、傾き、ゆるい曲線）」で表現するモデル
周期成分（sin）と非周期成分（trend）を同時にフィットする回帰モデル.  

	•	同定: sin + trend, 2成分同時

\begin{aligned}
u_U(s) &\approx a_U\cos ks + b_U\sin ks + c_U + d_U s (+e_U s^2),\\
u_V(s) &\approx a_V\cos ks + b_V\sin ks + c_V + d_V s (+e_V s^2),
\end{aligned}

(必要時に$s^2$採否)

	•	sin部分 → 周期性を表す: \sin(ks), \cos(ks) によって周期的な揺れや振動成分を表現。
	•	trend部分 → 背景変化（傾き・カーブなど）を表す: 1, s, s^2 などで、周期とは別のゆるやかな変化（トレンド）を補う。

## Cell 87

モデル式にはcosとsinの両方を入れる方が良い（＝安全）.   
1) cos だけ、または sin だけでは「位相の自由度」がない.  
   =>\cos(ks) だけ使うと、波の位相（ずれ） を表せない.  
   ex 真の信号が.  u(s) = A \sin(ks + \phi)とした場合にこれを展開すると.  
      u(s) = A\sin(ks)\cos\phi + A\cos(ks)\sin\phi.  
      ここで、   
      a = A\sin\phi, \quad b = A\cos\phi.  
      と置けば、   
      u(s) = a\cos(ks) + b\sin(ks).  
   つまり、cos と sin の両方を入れると、振幅と位相の両方を自由に表現できるようになる（sinだけやcosだけだと、波の始まり位置が固定されてしまう）.  
   u(s) = a\cos(ks) + b\sin(ks)は、u(s) = A\sin(ks + \phi)と等価.  
   ただし.  
   A = \sqrt{a^2 + b^2}, \quad \phi = \tan^{-1}\left(\frac{a}{b}\right).  

## Cell 88

トリムは共通化：DCTとsin+trendの両方に同じトリム（1 mm）を適応する。   
※Bishopではu_U, u_Vは全点で計算

## Cell 89

入力：端点を除去・デトレンド済みの u_U(s), u_V(s)、弧長 s（すべて voxel 単位）、候補波数リスト k_seeds（rad/voxel）、事前波数 k_prior。   
各候補 k について：    
1.	その近傍（±20〜30%）で微細グリッド探索（R² 最大化）。    
2.	[\,\cos ks,\ \sin ks,\ 1,\ s,\ (s^2)\,] を共通の設計行列にして u_U と u_V を同時に最小二乗（重み付き可）。   
3.	回転不変の振幅 A=\sqrt{a_U^2+b_U^2+a_V^2+b_V^2}、\lambda=2\pi/k、決定係数 R^2 を算出。   
4.	（任意）3次高調波比や事前からの乖離も計算してスコア化。   
•	出力：候補ごとの結果と、ベスト解（R² などで選抜）。   

## Cell 94

# grid_search_k(...)：最適な波数 k を「粗→細」で探す
目的：
観測した横変位 u_U(s), u_V(s) を
\begin{aligned}
u_U(s)&\approx a_U\cos(ks)+b_U\sin(ks)+c_U+d_U s~(+e_U s^2),\\
u_V(s)&\approx a_V\cos(ks)+b_V\sin(ks)+c_V+d_V s~(+e_V s^2),
\end{aligned}
で同時フィットしたときに一番よく合う波数 k を見つける。

## Cell 95

良いkのコンセプト：
1.	sinを足すことで “ちゃんと” 残差が減っている（≒周期性がある）.       => RG（Residual Gain）で評価.  
2.	全体としてもそこそこデータを説明している（R2が低すぎない）.          => R2で評価.  
3.	波形が“きれいな1周期の座屈”っぽい（hr指標が小さい）.               => hr（hr_amp / hr_energy）で評価.  
4.	理論的な λ_prior からはあまり外れていない（でも“正解”扱いはしない）. => λ_priorとの距離、で評価.  

R2だけだと「トレンドで説明したぶん」と「sin で改善したぶん」が区別できない.  

$ partial R^2(k) = 1 - \frac{SSE_{trend+sin}(k)}{SSE_{trend-only}} $.  
(https://best-biostatistics.com/toukei-er/entry/squared-partial-correlation-a-comprehensive-guide-from-concept-to-r-implementation/)
trend-only の残差 SSE_0 に対してsin+trend の残差 SSE_1(k) がどれだけ減ったかを見る量
= R2は「全体としてどれくらい説明しているか」、partial_R2は「sin 項がどれくらい意味のある改善をもたらしているか」なので、“周期性を探したい” という目的を評価できる.  

=> 「k を選ぶ問題」と「周期性あり／なしを判定する問題」は分けて考える.  
-> partial_R2は“周期性ありか？”の判定に使う.  
-> kの細かい選択はR2＋hr＋λ_prior でやる.  
※partial_R2をkを選ぶ際に使うと、coreでは問題ないが、Secondary/Straight（周期性なし）のセグメントではもともとpartial_R2が小さい・意味のあるピークがないため、「そもそもまともにフィットしない or 変なkが選ばれそう」

## Cell 102

# partial_R2 の閾値をどう決めるか
F検定を使用して求める
temporary noteの"9. "trend-only model"と"trend + sin model"の比較をpatial F検定で行う"を参照

$\boxed{
partial \,{R^2} =
\frac{F df_1}{F df_1 + df_2}}$

df_1 ＝ 第1自由度（分子の自由度）つまり 2.  
df_2 ＝ 第2自由度（分母の自由度）つまり N - 4.  

## Cell 105

ここで言う「ノイズ」には、   

MRA画像のボクセル解像度由来のギザギザ（0.5 mm ステップ）.  
セグメンテーション誤差・centerline 抽出の揺れ・座標変換/補間の誤差.  
frame（T,U,V）のわずかなjitter(信号の揺らぎ).  
モデル化しなかった高調波・細かい蛇行.  
本当にランダムな measurement noise.  

が全部まとめて入っている.  

## Cell 106

A のしきい値について.  

振幅 A は「信号の大きさ」なので、SNR = A_RMS / noise_RMS を考えると.  
SNR  0.5 未満 -> ノイズと同じかそれ以下.  
SNR  1.0 前後 -> ノイズと同じオーダー.  
SNR  2.0 以上 -> ノイズの 2 倍以上で、目視しても“揺れている”のがわかりやすい.  

=> SNR > 1.5を基準とする.  

A_hatが1 voxel(=0.5mm)未満の “揺れ”であった場合に、血管の形態とみなすか否か問題もあるので.  
A_hat > 1 voxel (0.5mm)の条件もつける

## Cell 109

Power_DCTグラフについて.  
sin+trend で当てたあと、どの周波数帯のエネルギーがどれくらい減ったかを大まかに知りたい.  
k_hat 付近（推定した波数）にあったパワーは、residual でちゃんと減っているか？   
残差のパワーは、orig と比べて全体的に何桁くらい小さくなっているか？   
まだ大きなピークが残っている周波数はあるか？（＝高調波や別モード）.  

## Cell 110

Power - DCT index m plot interpretation:   
高いmでは青とオレンジがほぼ重なっているのは正常.  
→ sin+trend モデルは「滑らかな 1 周期＋トレンド」だけなので、細かいギザギザは最初からモデル化していない.  
ここがピッタリ重なっているなら、   
「高周波成分は、フィット前後でほぼ変わっていない」つまり「座屈の 1 周期成分だけを主に取り除いた」

## Cell 112

線形代数の基本事実: 「円に任意の線形変換（2×2行列）をかけると、図形は一般に “楕円” になる」.  

sin+trend fittingの"sin成分だけ"をとりだすと.  
$$
h_U(s) = a_U\cos(ks) + b_U\sin(ks),\\
h_V(s) = a_V\cos(ks) + b_V\sin(ks)
$$
これをまとめて

$$
\mathbf{h}(s)
=
\begin{pmatrix}
h_U(s) \\
h_V(s)
\end{pmatrix}
=
\begin{pmatrix}
a_U & b_U \\
a_V & b_V
\end{pmatrix}
\begin{pmatrix}
\cos(ks) \\
\sin(ks)
\end{pmatrix}
=
M
\begin{pmatrix}
\cos(ks) \\
\sin(ks)
\end{pmatrix}.
$$

と書ける（Mは2×2の行列）。右側の$\begin{pmatrix}
\cos(ks) \\
\sin(ks)
\end{pmatrix}$は円なので、それに行列Mをかけると、「円を線形変換したもの」になる。つまり、線形変換された円=楕円。  

$h(s)$はU–V 平面の中で 楕円状の軌跡を描き、「横ずれベクトルの先端が、その平面内でどう動くか」を見ている.  
もし本物の 1D buckling（平面内で一方向だけに蛇行）なら.  
=> 「横ずれの向き」はほとんど変わらず、「大きさ（符号含む）」だけが変わる.  
=> つまり U–V 平面内でみるとほぼ 1 本の直線上を行ったり来たりする.  
もし 3D っぽくグネグネ回る（軸の周りを回るヘリックスなど）なら. 
=> 横ずれの向き自体もぐるぐる回る.  
=> U–V 平面内では 円に近い軌跡になる.  
2×2共分散行列の固有値比λ2/λ1で測っている、というのがdirectionality 

## Cell 116

# 真のピーク以外のDCTピークは何だったのか？

1.	有限長＋非整数周期 -> スペクトルリーク.  
→ 真の波数k0の周辺にエネルギーがにじみ出て複数の山になる(区間長Lに対してちょうど整数m周期入っていない場合、エネルギーがk0±Δkににじむ).  
2.	波形が歪んでいる -> 高調波（2k0, 3k0...）.  
→ 基本波の倍数近傍にもピークが立つ.  
3.	信号が多成分（buckling + 脳表地形 + 分岐 + ノイズ）.  
→ それぞれの空間スケールの成分が別々のピークとして現れる.  
4.	長波トレンドの取り残しピーク.  
→ λ_priorよりはるかに長い側に「C字全体」由来のピーク(多項式で取りきれなかった超長波成分、λ_priorよりかなり長いところにボコっと山を作る).  
5.	ノイズ＋有限サンプル.  
→ たまたま盛り上がった偽ピーク。uU/uVの残差RMSを「実効ノイズスケール」とみなしているので、SNRが低い候補ピークはほぼこのカテゴリと考えられる.   

このうち明示的にチェックする価値があるのは(2)高調波成分(3)多成分（別の物理モード）と場合によって(4)トレンドの取り残し.  

ただし、
DCT で候補抽出.  
各候補への sin+trend フィット（R2, partial R2, hr, coverage, SNRなどを評価）.  
ΔR2 <= 0.006 の Tie set + choose_k_from_tie_set (1. hr最小、2. 理論λに近い、3. より低いk(超波長)の順に採択) 
で、
1. リーク.  
2. 高調波.  
3. 多成分.  
4. トレンドの残り.  
5. ノイズ.  
のどれであっても、本命の buckling-like 周期だけを自然に選ぶ仕組みにはなっているが、もちろん例外があり、
L/λが1.0付近のLong-waveケースでは、「bucklingというよりC字トレンド」のピークが候補に入ってしまうこと.  
複数スケールのbucklingが混ざっている場合（Ex. 2 mode superposition）.  

本当に曖昧なな状況では、「本命だけを自然に選べる」とまでは言い切れないので.  
"Additional spectral peaks in the DCT power spectrum summed across both transverse components were interpreted as consequences of finite-length sampling (spectral leakage), higher harmonics, background tortuosity, or noise.

For each candidate peak within 0.5–2.0×λ_prior, we performed a joint sinusoid‑plus‑trend fitting to u_U(s) and u_V(s). Candidates whose coefficient of determination (R2) was within ΔR2 ≤ 0.006 of the best-fitting model were regarded as statistically equivalent (“tie set”, based on an AIC-equivalent threshold). Within this tie set, we designed to preferentially select buckling-like periodicity and selected the final wavenumber by minimizing, in order of priority, (i) the harmonic ratio (E3_/E1), (ii) the deviation of the observed wavelength from the theoretical scaling $\lambda \propto r$, and (iii) the wavenumber (preferring lower k as the fundamental mode)."

“Additional spectral peaks in the DCT power summed over the two transverse components were interpreted as consequences of finite-length sampling (spectral leakage), higher harmonics, background tortuosity, or noise.
For each candidate peak within 0.5–2.0×λ_prior, we performed joint sinusoid-plus-trend fitting.
Candidates with R² within ΔR²≤0.006 were considered statistically equivalent, and among them we selected the one with the lowest harmonic ratio (E₃/E₁), closest wavelength to the theoretical scaling λ∝r, and lower wavenumber, in this order.”

## Cell 117

# Coreはλ_hat・Aの主解析
λ_hat、Aの主解析.  
-=> Coreだけでやる（年齢・半径との関係など）.  

# Secondaryは“存在確率＋不確実性”を出す補助解析
Secondary セグメントは曲率が細かくは見えない（λは不明）.  しかし全体的な曲がり（macro-scale tortuosity）はしっかり含まれる.  
=> 「Secondary は λ が分からないだけで、伸長 (ρ)、曲率、planarity という形態変数の点では Core と同格のデータ」.  

※
λ を使う解析はCore限定.  
それ以外の形態（elongation/tortuosity/曲率）を見る解析にはCore + Secondary を両方使うのが筋が良い.   

よって.  

→ Secondary からも年齢効果の主要部分（伸長、余剰長、曲率の増加）を強力に拾える.  
※ 階層モデル（Hierarchical Bayes）の partial pooling に使える

### ベイズで Secondary を救う案は理論的には面白いが、初回の論文としては「重い＆誤解されやすい」ので、まずは使わずにシンプルなカテゴリ扱いにしておくのが良さそう。


# Long-wave / Straight-ish / undetectable
※ポイント：“短いから見えないだけ” なものと、「本当に波がない」ものは理論的には区別不能 = 「今回の解析フレームでは buckling モードを同定できなかった」群。論文では、は「この測定長・分解能のもとでは buckling 型周期を検出できなかったセグメント」と operational に定義すると書いておくのが正直か。






### Main analysis の柱（論文の「結果」）

TooShort / Unresolved: 解析母集団から外す（分母に入れない）, 「検出不能」なだけで「真にまっすぐ」とは言えないから.  

「buckling の発現段階」として混合ロジスティック/順序ロジスティックで見る.  
(1) λ と半径 r のスケーリング（buckling モデルのチェック）.  
 対象：Core_final（かつ半径を適当なレンジで絞る）.   
 回帰：$\log \lambda_{hat} = \beta_0 + \beta_r \log r + \beta_{age} \cdot \text{age} + \dots$.   
 期待：   
 	$\beta_r \approx 1（λ ∝ r）$.  
	$\beta_{\text{age}} \approx 0$（年齢依存は弱い）.  
	シンプルに見るなら：
	各年齢帯ごとに $\lambda_{hat} / r$ の分布を描いて、年齢で大きく変わっていない（だいたい一定）ことを示す.  
 追加：$\log (\lambda_{hat} / r)$を目的変数にして$\log (\lambda_{hat} / r)$ ~ age + 部位 + (1|case)もやっておくと、「$\log (\lambda / r)$ が年齢でほぼ一定」という形で示せてきれい

(2) 振幅 A の年齢依存（“buckling が強くなる”部分）.   
 対象：Core_final.  
 指標：
	$A [mm]$ あるいは無次元化 $A / L$.  
 解析(例)：   
	$\log(A/L) = \gamma_0 + \gamma_{age} \cdot \text{age} + \gamma_r\log r + \dots$.  
 期待：$\gamma_{age} > 0$
=>「波長は (ほぼ) 半径で決まり、年齢ではあまり変わらないが、振幅は年齢とともに増大する」という物理モデルと整合的かどうかをチェック。bucklingモデルの“教科書的挙動”と整合   

(3) 存在確率（Straight / Secondary / Core の割合 vs 年齢）.  
 従属変数：   
	“buckling あり” = Core_final（＋Secondary を含めるかどうかは要検討、感度解析として “Core+Secondary vs Straight-ish” も一応見る?）.  
	vs Straight-ish.  
 解析(例)：
	ロジスティック回帰：$\Pr(\text{bucklingあり}|\text{age},r,\text{部位})$.  
	•	年齢が上がると buckling を持つセグメントの割合が増えるかどうかを見る.  
 イメージ: 各セグメントが「buckling 波が検出レベルまで育っているかどうか」という二値の“検出イベント”.  

 これを年齢・半径などの関数として見ると「年齢が上がると “検出されるセグメントの割合” が増えるか？」「太いほど・細いほど buckling が検出されやすいか？」といった“検出可能なレベルに育った buckling の頻度”を見ていることになる。
 =この割合にはこういう意味がある「ある年齢・血管径の条件のもとで、“buckling様の周期変位が、検出可能なくらいまで発達しているセグメント”の割合」を見ている、と解釈できる。

 これは、理論の観点から言うと、座屈モデルでは「負荷（年齢・血圧・組織変性など）が増えると buckling モードの振幅が徐々に増える」
 = その結果として 「検出可能なレベルまで育ったセグメント」が増えていくはずという予言に対応。

 若年では検出割合が低い・高齢になると検出割合が有意に増える、と出れば、「加齢とともに、buckling 的な周期変位が“検出可能なレベル”まで達しているセグメントが増える」という、現実的に観測可能なレベルの主張ができる

 1.	存在確率（有無）.  
セグメント単位: B_ij（buckling 検出有無） ~ 年齢 + r + 部位 + (1|被験者).  
=> 年齢で検出オッズ↑を示す、あるいは被験者単位で p_i（割合） vs 年齢.  
2.	強さ（振幅）：B_ij=1 のセグメントに限定して：log(A_true/L) ~ 年齢 + r + 部位 + (1|被験者).  
=> 年齢で振幅↑を示す.   
3.	波長の年齢非依存性（補助）.  
B_ij=1 のセグメントに限定して：log(λ_hat/r) ~ 年齢 + 部位 + (1|被験者).  
=> 年齢効果 ~0 を示し、「波長は主にrで決まる」を確認.  

※「ゼロが多い λ_hat, Aをそのまま混合モデルに入れる」のではなく、“有無” と “強さ” を分けて扱うのがポイント.  

"bukcling存在有無"と"強さ(A/L)"を分ける設計としている.  
PDE的には「臨界を超えると A が立ち上がる」ので、   
•B_ij：臨界超えたか？（検出レベルまで育ったか？）.  
•A/L：超えた後どれだけ進んだか？   
という二層構造で解釈できる！   


##### 追加でやると良さそうな解析(1)
bucklingモデル的には、臨界伸長P_cを超えるとAが立ち上がるので、   
Core＋Secondary で.  
• rho vs A/L（Coreのみ）.  
•または rho vs buckling有無（Core/Secondary vs Straight）.  
加齢を共変量に入れて「rho が大きいほど buckling 検出確率が高いか」「同じrhoであれば、年齢でA/Lに差が出るか」みたいな解析があると、「加齢と伸長のどちらが効いているか」を切り分けるヒントになりそう.  

##### 追加でやると良さそうな解析(2)
planarity / A_U vs A_V で「平面内 buckling」の確認.  
物理モデル的には、columnは主にある一つの平面内に座屈する.  
よってCoreセグメントで.  
•A_U vs A_V の比（例えば A_U/A_V の分布）.  
•planarity指標との関係.  
を見ると、「Coreのbuckling は、主に局所的な平面内で起きている」（= 真のbucklingっぽく、3Dランダム蛇行ではない）ことの裏付けになる。
!注意!   
1. bucklingが“ほぼ1枚の平面の中で起きているか？”（＝1D columnのモデルが妥当か？　この場合、どの平面内なのかは問わない）.  
   => 1D buckling モデルを数理的に適用する」には“ほぼ平面内”であれば十分（平面の向きは任意）.  
   さらに細かく分けると
   	1.1.	globalな幾何（original centerlineがどれくらい平面っぽいか）
	1.2.	buckling モード（特定のk_hat成分）が1D的かどうか
2. その平面が “脳表にだいたい平行か？”   
   => 「解剖学的に説得力を持たせたい（脳表に沿っているはず）」=“その平面が脳表におおむね平行”であると見せたい
は別で考え、証明する必要がある。   
1については
$r_{minor} = \frac{\min(|A_U|,|A_V|)}{\max(|A_U|,|A_V|)}$
のような「minor/major振幅比」を見て、Coreセグメントで$r_{minor}$が一貫して小さい（<<1）なら「Bishopの法線平面の中でも、実際のbuckling変位はほぼ1方向に偏っている」-> 3Dランダム蛇行ではなく、ほぼ1D的な“平面 buckling”として扱える。ここでの脳表との位置合わせは不要。   
またPCAから算出したplanarity指標(λ2−λ3)/λ1が高い(planarity>>0)なら、中心線全体としても、Core はほぼ2Dに埋まっている（空間的に“薄い板”上）という裏付けになる

3. 改良!! 1.1と1.2についての考察！
planarity：original centerlineがそもそも2Dに近いか？（global）.  
directionality：その面内で、buckling モードが「ほぼ 1 本の線」か？（local mode）.  
    sin+trendのsin部分だけ を取り出して
    $h_U(s) = a_U\cos(ks) + b_U\sin(ks)$
    $h_V(s) = a_V\cos(ks) + b_V\sin(ks)$
	その2D ベクトル$h(s) = (h_U(s), h_V(s))$の点群（sを動かしたときの軌跡）から 2×2 共分散行列を作り、その固有値比.  
    $ directionality = \lambda_2 / \lambda_1$.  
    これは推定したbucklingモード（波長 λ_hatの成分）が、法線平面の中でどれくらい“1方向”に揺れているか？の指標になるのではないか.  
    λ₂/λ₁ << 1 -> ほぼ1本の線上 -> このモードは1D的.  
	λ₂/λ₁ $\approx$ 1 -> 円に近い -> このモードは2D的に散らばっている.   

 
##### 追加でやると良さそうな解析(3)
被験者レベルへ集約した解析を行うと「人単位の指標」としてわかりやすいかも.  
セグメント単位だけでなく、被験者ごとに.  
•Core セグメント数 / 全セグメント数（検出率）.  
•Core の平均 A/L.  
•最大 A/L など.  
を集約して、「被験者単位で、年齢 vs 『buckling の程度』」を見る簡単な回帰も1つ入れておく?


## 注意：
太くて短い血管(M1など)は、物理的に「1.5周期以上のbuckling波」を載せるだけのスパンはない.  
-> ちゃんとした正弦モードとして推定するのがそもそも難しい.  
-> Core 条件から自然にこぼれ落ちる.  
「太くて短い近位主幹は、自由に座屈モードを発達させるにはスパンが短すぎ、buckling モデルによる波長・振幅推定の対象外」と明記する!   


これは今回の解析上の問題点というより、   
「buckling モードとして定量できるような“自由スパン”がそもそも存在しない」.  
という幾何学的な制約の反映と考える.  
(“バイアス” というより “モデルが適用できる領域の限定” ).  


フィットしそうなまとめは：   
Coreが中〜末梢セグメント寄りになるのは、かなり不可避.  
これは「偏り」というより「buckling モードを1.5周期以上きちんと観測できる自由スパンが、もともとそういう部位に限られている」という幾何学的事実.  
主解析では：    
対象半径レンジ（例：0.4–1.2 mm）と対象部位（M2–3, M4 など）.  
を明示的に “解析対象” として宣言する.  
=> 近位・太い trunk（M1 等）は、buckling モード解析の対象外であることを明記し、必要なら別枠で形態的特徴（C 字変形など）だけを定性的に示す、という「サブ解析」として扱い

## Cell 118

脳表との位置合わせなどを利用したデータでない場合（つまり今やり方の場合）、sin+trend fittingが血管の座屈による波ではなくて脳表の地形的な起伏が実は拾われている可能性はある!   

基本的には脳表の地形のかなり長い起伏はP0側に押し込んであるという設計 +  Long-wave / Straight-ish 判定側で処理される.  
その上で血管自身の細かめな蛇行（半径の数倍スケール）はDeltaの高周波成分としてuU/uVに載ってくるはずなので、そこに sin+trend を当ててλ_hat, A_hatを出している. 
という「スケールの分け方」をしている.  

ただし、   
セグメントが短くて脳表に沿った1回の曲がりと buckling 的な起伏が同じスケールになってしまうとき. 
や.  
脳表の地形そのものが「割ときれいな周期性（脳回・脳溝が等間隔）」を持っていて、かつ血管がそこに忠実に沿って走っているとき.  
は
数学的には「脳表に沿ってクネクネしている」のか「座屈している」のか中心線の形状だけでは区別不能.  
=> よって、形状だけ見て物理メカニズムを「証明」しているわけではなく、「観測されたλ, A, そのr依存性がbucklingモデルと整合しているか」をチェックするという流れ!!
