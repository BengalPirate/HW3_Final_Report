# Git Push Instructions for HW3 Final Report

## Step-by-Step Commands

Open your terminal and execute these commands in order:

### 1. Navigate to the HW3 directory
```bash
cd /home/x/Desktop/HW3_Group_Project/HW3
```

### 2. Initialize Git repository (if not already done)
```bash
git init
```

### 3. Create .gitignore file (to exclude data files)
```bash
cat > .gitignore << 'EOF'
# Data files (per assignment instructions - do not include)
trainingData.txt
trainingTruth.txt
testData.txt
blindData.txt

# Output files (optional - remove if you want to exclude these too)
# testLabel_lightgbm.txt
# blindLabel_lightgbm.txt
# testLabel_confidence.txt
# blindLabel_confidence.txt

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Virtual environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
EOF
```

### 4. Add the remote repository
```bash
git remote add origin git@github.com:BengalPirate/HW3_Final_Report.git
```

### 5. Check your git configuration (optional but recommended)
```bash
git config user.name "Brandon Newton"
git config user.email "your-email@example.com"  # Update with your actual email
```

### 6. Add all files to git
```bash
git add .
```

### 7. Check what will be committed (optional - good practice)
```bash
git status
```

You should see:
- ✅ FINAL_REPORT.md
- ✅ SUBMISSION_SUMMARY.md
- ✅ Enhanced_lightbgm.py
- ✅ lightbgm.py
- ✅ enhanced_lightgbm_blinddata.ipynb
- ✅ testLabel_lightgbm.txt
- ✅ blindLabel_lightgbm.txt
- ✅ Phase1_Alternative_Methods/ (entire folder)
- ❌ trainingData.txt (should be ignored)
- ❌ testData.txt (should be ignored)
- ❌ blindData.txt (should be ignored)

### 8. Commit the changes
```bash
git commit -m "Initial commit: HW3 Final Report and Code

- Final report with Phase 1 and Phase 2 approaches
- Phase 2 implementation achieving 92.82% test accuracy
- Phase 1 exploratory work (92% accuracy)
- All code files and documentation
- Test and blind predictions included

Team: Brandon Newton & Venkata Lingam
Course: CSC 621 Machine Learning"
```

### 9. Set the default branch to main (if needed)
```bash
git branch -M main
```

### 10. Push to GitHub
```bash
git push -u origin main
```

---

## Troubleshooting

### Issue: "Permission denied (publickey)"
**Solution:** You need to set up SSH keys with GitHub

```bash
# Check if you have SSH keys
ls -la ~/.ssh

# If no id_rsa or id_ed25519, generate new SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add SSH key
ssh-add ~/.ssh/id_ed25519

# Copy the public key to clipboard
cat ~/.ssh/id_ed25519.pub

# Then add this key to GitHub:
# 1. Go to GitHub.com → Settings → SSH and GPG keys
# 2. Click "New SSH key"
# 3. Paste the key and save
```

### Issue: "Repository not found"
**Solution:** Make sure the repository exists on GitHub

```bash
# Check remote URL
git remote -v

# If wrong, update it
git remote set-url origin git@github.com:BengalPirate/HW3_Final_Report.git
```

### Issue: "Repository already exists" when pushing
**Solution:** The remote repo might not be empty

```bash
# Pull first, then push
git pull origin main --allow-unrelated-histories
git push -u origin main

# OR force push (WARNING: overwrites remote)
git push -u origin main --force
```

---

## Alternative: Using HTTPS instead of SSH

If SSH is causing issues, you can use HTTPS:

```bash
# Remove SSH remote
git remote remove origin

# Add HTTPS remote
git remote add origin https://github.com/BengalPirate/HW3_Final_Report.git

# Push (will prompt for GitHub username and password/token)
git push -u origin main
```

**Note:** For HTTPS, you'll need a GitHub Personal Access Token (not your password):
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use this token as your password when prompted

---

## Quick Reference: What's Being Pushed

```
HW3/
├── FINAL_REPORT.md (1,450 lines)
├── SUBMISSION_SUMMARY.md
├── GIT_PUSH_INSTRUCTIONS.md (this file)
├── Enhanced_lightbgm.py (92.82% accuracy)
├── lightbgm.py
├── enhanced_lightgbm_blinddata.ipynb
├── testLabel_lightgbm.txt (13,082 rows)
├── blindLabel_lightgbm.txt (31,979 rows)
├── testLabel_confidence.txt
├── blindLabel_confidence.txt
├── README.md
├── .gitignore (excludes raw data)
│
└── Phase1_Alternative_Methods/ (18 Python files)
    ├── README.md (2,100 lines)
    ├── FILE_GUIDE.md
    ├── ultimate_pipeline.py
    ├── fast_xgboost.py
    ├── optimized_pipeline.py
    └── (15 other files)
```

**Total:** ~30 files, ~5,000 lines of code, ~4,000 lines of documentation

---

## After Successful Push

Your repository will be available at:
**https://github.com/BengalPirate/HW3_Final_Report**

You can verify by:
1. Going to the URL above
2. Checking that all files are present
3. Viewing the FINAL_REPORT.md directly on GitHub

---

## Need Help?

If you encounter any issues:
1. Check the error message carefully
2. Try the troubleshooting steps above
3. Ensure you have write access to the repository
4. Verify SSH keys are set up correctly

**Last Updated:** November 11, 2025
