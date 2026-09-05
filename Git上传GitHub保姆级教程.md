# 🚀 NNProject 上传 GitHub 零基础保姆级教程（从本地到云端）

> **项目名称**：`NNProject`（基于 THUCNews 的新闻文本分类与深度学习系统）  
> **本地路径**：`C:\Users\asus\PycharmProjects\NNProject`  
> **GitHub 用户名**：`Liu`  
> **目标云端地址**：`https://github.com/Liu/NNProject.git`  
> **适合人群**：已完成本地开发，想把当前项目完整、规范、安全地推送到 GitHub 的同学。你可以一边看这篇文档，一边在 PyCharm 终端里“慢慢弄”。

---

## 目录
1. [五分钟搞懂：Git 与 GitHub 的核心模型（NNProject 实战版）](#一五分钟搞懂git-与-github-的核心模型nnproject-实战版)
2. [第一步：身份与配置快速确认（你已搞定，自检即可）](#二第一步身份与配置快速确认你已搞定自检即可)
3. [第二步：在 GitHub 网页上新建远程仓库（⚠️避坑必读）](#三第二步在-github-网页上新建远程仓库避坑必读)
4. [第三步：关联远程仓库地址（已为你校准）](#四第三步关联远程仓库地址已为你校准)
5. [第四步：深度学习项目避坑（.gitignore 文件守护）](#五第四步深度学习项目避坑gitignore-文件守护)
6. [第五步：本地打包与提交存档（git add & commit）](#六第五步本地打包与提交存档git-add--commit)
7. [第六步：推送到 GitHub（第一次 Push 与登录鉴权）](#七第六步推送到-github第一次-push-与登录鉴权)
8. [第七步：大功告成！在 GitHub 网页端验收成果](#八第七步大功告成在-github-网页端验收成果)
9. [第八步：在 PyCharm 中纯鼠标操作（日常开发神器）](#九第八步在-pycharm-中使用纯鼠标操作日常开发神器)
10. [第九步：以后的日常三步曲与时光机后悔药](#十第九步以后的日常三步曲与时光机后悔药)
11. [NNProject 专属命令速查表](#十一nnproject-专属命令速查表)

---

## 一、五分钟搞懂：Git 与 GitHub 的核心模型（NNProject 实战版）

在开始敲命令前，花 2 分钟建立直观心智模型，后面每一步操作你就都会知道“自己在干什么”：

```
+---------------------------+      git add .      +---------------------------+     git commit -m "..."     +---------------------------+      git push origin main      +-----------------------------------------+
|        1. 工作区          | ------------------> |         2. 暂存区         | ------------------------> |        3. 本地仓库        | -----------------------------> |               4. 远程仓库               |
|      (Working Tree)       |                     |          (Index)          |                           |    (Local Repository)     |                                |           (Remote: GitHub)              |
+---------------------------+                     +---------------------------+                           +---------------------------+                                +-----------------------------------------+
  你在 PyCharm 里看到的代码：                          准备好“打包装箱”的文件列表：                                本地固若金汤的时光机存档：                                        全世界/云端备份的 GitHub 仓库：
  - NN/train.py                                       （排除了 .zip、权重、缓存）                                 （你电脑断网、关机都丢不了）                                     github.com/Liu/NNProject
  - 飞书md/综合课程设计.md                                                                                        已包含第一次 commit：                                            （让面试官、老师或合作者查看）
  - 第一小组课程设计报告.pdf                                                                                      "Initialize THUCNews..."
```

- **工作区**：你当前 PyCharm 里的所有文件。
- **暂存区**：准备好提交的清单（类似快递打包盒）。
- **本地仓库**：你电脑里的 `.git` 文件夹。每次 commit 相当于拍了一张完整快照（存档点）。
- **远程仓库（GitHub）**：云端托管中心，负责云端备份和展示。

---

## 二、第一步：身份与配置快速确认（你已搞定，自检即可）

你在提问中提到 **`username` 和邮箱已经 config 成功**。  
我们已经在底层为你核实过了，你的当前配置如下：
- `user.name` = `Liu`
- `user.email` = `2534985567@qq.com`

你可以打开 PyCharm 下方的 **Terminal（终端）**，输入以下命令自己看一眼：

```bash
git config user.name
git config user.email
```
* 如果分别输出了 `Liu` 和你的邮箱，就说明配置百分之百正确，**无需重复设置**，直接进入下一步！

---

## 三、第二步：在 GitHub 网页上新建远程仓库（⚠️避坑必读）

请打开浏览器，按以下步骤在 GitHub 上建一个空仓库：

1. 打开 [https://github.com](https://github.com) 并登录你的账号 **`Liu`**。
2. 点击网页右上角的 **`+`** 号，选择 **`New repository`**（新建仓库）。
3. 填写仓库信息：
   - **Repository name**（仓库名）：必须填 **`NNProject`**（大小写建议完全一致）。
   - **Description**（项目简介，选填）：例如 `基于预训练词向量与深度神经网络的中文新闻文本分类系统`。
   - **Public / Private**：
     - 如果用于课程作业展示、个人简历加分，推荐选 **Public**（所有人可见）；
     - 如果暂时不想公开，选 **Private**（仅自己可见）。
4. ⚠️ **最关键的一步（新手 99% 会踩的坑，千万注意！）**：
   - ❌ **不要勾选** `Add a README file`
   - ❌ **不要勾选** `Add .gitignore`
   - ❌ **不要勾选** `Choose a license`
   > **为什么不要勾？**  
   > 因为你的本地 `NNProject` 已经有了 `README.md` 和 `.gitignore` 文件，并且已经有了提交记录。如果你在网页端勾选了这些，GitHub 会自动在线生成一个新的不同提交，导致**两端历史冲突**，初学者会遇到复杂的 merge/rebase 报错。保持一个**完全干净的空仓库**是最省心的！
5. 点击最下方的绿色按钮 **`Create repository`**。
6. 创建成功后，GitHub 页面会显示你的专属仓库链接：
   `https://github.com/Liu/NNProject.git`

---

## 四、第三步：关联远程仓库地址（已为你校准）

本地项目需要知道该把代码推送到 GitHub 的哪个具体网址。

之前本地仓库的地址里写的是占位符 `你的用户名`，**我们刚才已经帮你执行命令校准为了 `Liu`**。  
你可以亲自在终端输入以下命令验证：

```bash
git remote -v
```

* **预期正确输出**：
  ```text
  origin  https://github.com/Liu/NNProject.git (fetch)
  origin  https://github.com/Liu/NNProject.git (push)
  ```

> 💡 **备忘知识**：如果以后换了仓库名字，只需敲这行命令就能随时更改：
> ```bash
> git remote set-url origin https://github.com/Liu/NNProject.git
> ```

---

## 五、第四步：深度学习项目避坑（.gitignore 文件守护）

在机器学习与深度学习项目中，有一个**极危险的操作**：
> ⚠️ **千万不要把几十 MB 的压缩包（如 `AtoC.zip` 51MB）或者几百 MB 的模型权重（`.pth`, `.npy`）推送到 GitHub！**  
> GitHub 单个文件超过 100MB 会直接阻断推送；即便只有 50MB，也会导致上传极慢、消耗网络流量。

在你的项目根目录下，已经为你配置好了 `.gitignore` 文件，它会自动忽略：
1. `AtoC.zip`（51MB 的压缩包，安全排除！）
2. `**/dataset/*.txt`、`*.clean.txt`（大型原始文本文件）
3. `*.pth`、`*.model`、`*.npy`（训练好的神经网络权重和向量缓存）
4. `.idea/`（PyCharm 个人偏好配置）
5. `__pycache__/`（Python 运行时临时缓存）
6. `*.aux`, `*.log`, `*.out`, `*.toc`（刚刚为你补充添加的 LaTeX 编译中间临时文件）

这样，最终被推送到 GitHub 的将是**高质量的代码、说明文档、报告 PDF 和图表**，非常清爽专业！

---

## 六、第五步：本地打包与提交存档（git add & commit）

现在我们把本地新添加的文档、报告和修改后的配置文件进行打包和正式存档。

### 1. 查看当前状态
在终端输入：
```bash
git status
```
你会看到一些红色提示的文件（包括修改过的 `.gitignore`，以及新增的飞书文档、报告 PDF 等）。

### 2. 将所有文件加入暂存区（装箱）
在终端输入：
```bash
git add .
```
> （注意 `add` 后面有一个空格和一个英文句号 `.`，代表当前目录下的所有有效改动）

再次输入 `git status`，你会发现原本红色的文件全部变成了**绿色**，说明装箱成功！

### 3. 提交本地存档（盖章入库）
在终端输入：
```bash
git commit -m "feat: 完成新闻文本分类报告与数据分析脚本"
```
* 提示：`-m` 后面引号里的内容是提交说明，清晰描述这次提交了什么。

### 4. 确保主分支名称为 main
```bash
git branch -M main
```
* （当前本地分支默认已经是 `main`，执行这行能确保 100% 与 GitHub 默认标准分支吻合）

---

## 七、第六步：推送到 GitHub（第一次 Push 与登录鉴权）

这是最激动人心的一步，也是新手最容易遇到弹窗的一步，别慌，跟着下面做：

在终端输入推送命令：
```bash
git push -u origin main
```
* `-u origin main` 表示：将本地的 `main` 分支推送到远程 `origin`，并建立跟踪关系。以后再推送只需要敲简短的 `git push` 即可。

---

### 🚨 常见弹窗与登录情况应对（对号入座）：

#### 情况 A：弹出 Windows 凭证管理器网页授权（最常见、最便捷）
因为你的 Git 已开启凭证管理器（`credential.helper=manager`），终端可能会自动弹出一个浏览器的 GitHub 授权页面：
1. 页面提示：**`Sign in with your browser`**；
2. 点击绿色的 **`Authorize git-ecosystem`** 按钮；
3. 浏览器提示 Success，终端立刻开始上传并显示进度（`Writing objects: 100%...`）；
4. **大功告成！** 凭证管理器会永久记住你的电脑，以后再推送都不会再弹窗。

---

#### 情况 B：终端要求输入 Username 和 Password
如果在终端黑框里提示输入 `Username for 'https://github.com':` 和 `Password for '...':`：
1. **Username** 填：`Liu`（或你绑定的邮箱 `2534985567@qq.com`）；
2. **Password**：⚠️ **注意！这里不能输入你平时的 GitHub 网页登录密码！**  
   GitHub 早已废弃了密码上传，必须使用 **Personal Access Token（个人访问令牌）**。

> **手把手 1 分钟获取 Token（如果遇到了才需要做）：**
> 1. 打开 GitHub，点击右上角个人头像 -> 点击 **Settings**。
> 2. 页面拉到最左下角，点击 **Developer Settings**。
> 3. 点击 **Personal access tokens** -> **Tokens (classic)**。
> 4. 点击右上角 **Generate new token** -> **Generate new token (classic)**。
> 5. **Note**（备注）随便填，比如 `MyPC`；
> 6. **Expiration**（有效期）推荐选 `90 days` 或 `No expiration`；
> 7. **Select scopes**（权限）：把第一个 **`repo`** 打上勾（代表拥有推送代码权限）；
> 8. 滑到最底部，点击绿色的 **Generate token**；
> 9. 页面会显示一串以 `ghp_` 开头的长字符。**立刻点击右侧按钮复制它**（刷新后就看不到了）；
> 10. 回到终端，在 `Password:` 提示处**粘贴刚才复制的这串 Token**（注意：终端里粘贴密码通常是不显示任何字符的，直接按回车即可）。

---

#### 情况 C：国内网络连接超时（Failed to connect to github.com port 443）
如果报错提示连接不上 GitHub：
- **原因**：国内网络偶尔访问 GitHub 不是很稳定。
- **应对方案**：
  1. 如果你有科学上网代理软件（例如端口是 7890），可在终端执行：
     ```bash
     git config --global http.proxy http://127.0.0.1:7890
     git config --global https.proxy http://127.0.0.1:7890
     ```
  2. 如果关闭了代理软件，记得清空代理配置：
     ```bash
     git config --global --unset http.proxy
     git config --global --unset https.proxy
     ```
  3. 或者直接尝试手机热点连接电脑，往往能秒通！

---

## 八、第七步：大功告成！在 GitHub 网页端验收成果

推送成功后，终端会打印类似如下信息：
```text
Enumerating objects: ...
Counting objects: 100% ...
Writing objects: 100% ...
To https://github.com/Liu/NNProject.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

此时，打开浏览器访问你的项目主页：
👉 **`https://github.com/Liu/NNProject`**

刷新页面，你会看到：
1. 所有的源码（`NN/`、`AtoC/src/`）、分析脚本和报告都整整齐齐地陈列在上面；
2. 首页自动渲染了项目的根目录 [README.md](file:///c:/Users/asus/PycharmProjects/NNProject/README.md)，排版极其专业；
3. 大体积的 `AtoC.zip`、数据集和无用编译缓存都没有被传上去，仓库轻巧健康！

---

## 九、第八步：在 PyCharm 中使用纯鼠标操作（日常开发神器）

第一次成功推送到 GitHub 后，**恭喜你！以后的日常开发中，你几乎不需要再敲任何黑框命令行了！**

因为你用的是专业 IDE **PyCharm**，一切操作都可以用鼠标完成：

```
+-------------------------------------------------------------------------------+
|  PyCharm 顶部菜单:  Git -> Commit... (快捷键 Ctrl + K)                         |
|                                                                               |
|  [✓] 勾选改动的文件 (如 train.py)                                               |
|  [ Commit Message 框 ]: 输入修改说明，如 "优化了 BiLSTM 的注意力机制"             |
|                                                                               |
|  点击右下角 Commit 按钮右侧的小三角 ▼                                           |
|  选择: 【Commit and Push...】  <--- 一键保存到本地并自动推送到 GitHub！          |
+-------------------------------------------------------------------------------+
```

1. **一键提交并推送**：快捷键 `Ctrl + K`（Mac 上是 `Cmd + K`），写完备注直接点 **Commit and Push**，一次性解决！
2. **可视化版本树**：点击 PyCharm 最左侧或最下方的 **Git** 标签页，所有提交历史、分支图全彩色展示，随时查看改动对比。
3. **撤销单行改动**：在编辑器里，被修改的代码行左侧会有淡绿色/淡蓝色竖条，鼠标点击竖条即可一键回滚该行。

---

## 十、第九步：以后的日常三步曲与时光机后悔药

如果你更喜欢在命令行敲击的极客感觉，以后的日常维护只需要牢记**极简三步曲**：

### 1. 每次写完一段代码后的标准推送流程
```bash
git status                     # 1. 看一眼改了哪些文件
git add .                      # 2. 全部打包进箱子
git commit -m "更新说明文字"     # 3. 盖章保存到本地历史
git push                       # 4. 同步到 GitHub
```

### 2. 紧急后悔药（写错代码如何撤销）
- **场景 1**：某文件（如 `train.py`）写乱了，还没提交，想瞬间恢复到上次保存的状态：
  ```bash
  git restore NN/train.py
  ```
- **场景 2**：想做一个危险的大实验（比如彻底重构模型架构），但怕改废原有代码：
  ```bash
  git checkout -b experiment_model   # 创建并进入一个平行的实验分支
  # 在这里随意尝试、随意 commit...
  # 如果实验成功：
  git checkout main
  git merge experiment_model         # 合并成果回主干
  # 如果实验彻底搞砸：
  git checkout main
  git branch -D experiment_model     # 挥一挥衣袖删除分支，主代码毫发无损！
  ```

---

## 十一、NNProject 专属命令速查表

| 操作需求 | 终端命令 | 说明 |
| :--- | :--- | :--- |
| **查看项目当前状态** | `git status` | 随时敲，看文件红绿状态，最重要命令 |
| **装箱所有变动** | `git add .` | 把当前修改放入暂存区 |
| **本地提交保存** | `git commit -m "提交说明"` | 写入本地版本历史快照 |
| **推送到远程** | `git push` | 同步本地提交至 GitHub（初次使用 `git push -u origin main`） |
| **查看提交历史** | `git log --oneline` | 单行简短查看版本时间轴 |
| **查看具体代码改动** | `git diff` | 查看未暂存的修改增删细节 |
| **放弃工作区某文件修改** | `git restore <文件名>` | 瞬间还原文件（后悔药） |
| **查看关联的远程仓库** | `git remote -v` | 确认是否关联到了 `Liu/NNProject` |
