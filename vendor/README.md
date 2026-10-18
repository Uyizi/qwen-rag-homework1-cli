# 终端界面离线依赖

这些纯 Python wheel 从本机已安装的发行包整理
没有下载文件，也不包含模型权重
各依赖的许可证保留在 wheel 内的 dist-info 目录中

| 依赖 | 版本 | 许可 |
| --- | --- | --- |
| Rich | 15.0.0 | MIT |
| markdown-it-py | 4.2.0 | MIT |
| mdurl | 0.1.2 | MIT |
| prompt-toolkit | 3.0.53 | BSD 3 Clause |
| wcwidth | 0.8.2 | MIT |
| Pygments | 2.20.0 | BSD 2 Clause |

本目录不包含 NumPy、PyMuPDF 或 CUDA 后端依赖
它们需要已有环境或另行准备匹配平台的离线 wheel
