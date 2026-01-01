# 贡献指南

## 如何备份你的本地文件

### 方法一：直接复制

1. 将你的本地 WEB-ROBOT 文件夹中的所有文件复制到这个仓库目录
2. 使用 Git 命令提交更改：

```bash
git add .
git commit -m "描述你的更改"
git push
```

### 方法二：使用 Git 同步

如果你的本地文件夹已经是一个 Git 仓库：

1. 添加远程仓库：
```bash
git remote add origin https://github.com/PB-in-GH/WEB-ROBOT.git
```

2. 推送到远程：
```bash
git push -u origin main
```

### 最佳实践

1. **定期备份**：建议定期将本地更改推送到 GitHub
2. **写好提交信息**：清楚描述每次提交的内容
3. **检查敏感信息**：确保不要提交密码、API密钥等敏感数据
4. **使用分支**：对于实验性功能，建议使用分支开发

## 常见问题

### Q: 如何恢复到之前的版本？

```bash
git log  # 查看提交历史
git checkout <commit-id>  # 切换到特定版本
```

### Q: 如何忽略某些文件？

编辑 `.gitignore` 文件，添加你想忽略的文件或目录。

### Q: 文件太大无法上传怎么办？

GitHub 对单个文件有 100MB 的限制。对于大文件，建议：
- 使用 Git LFS (Large File Storage)
- 将大文件存储在其他地方，仅在仓库中保留引用
