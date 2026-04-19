#!/usr/bin/env bash
# ============================================================
# Step 0: 环境安装脚本
# 在 conda base 环境下安装所有依赖
# 用法: bash reproduce/00_setup_env.sh
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
INSTALL_MINERU="${INSTALL_MINERU:-false}"
INSTALL_DOCLING="${INSTALL_DOCLING:-false}"
echo "=== RAG-Anything 环境安装 ==="
echo "仓库路径: $REPO_ROOT"
echo ""

if [[ ! -x "$VENV_PYTHON" || ! -x "$VENV_PIP" ]]; then
  echo "[ERROR] 未找到可用虚拟环境: $REPO_ROOT/.venv"
  echo "请先运行: python3 -m venv .venv"
  exit 1
fi

# ---- 1. 安装项目依赖（开发模式）----
echo "[1/4] 安装核心开发依赖..."
"$VENV_PIP" install \
  "setuptools>=64" \
  wheel \
  python-dotenv \
  pytest \
  openai \
  tqdm \
  huggingface_hub \
  lightrag-hku

echo "[2/4] 安装 raganything（editable 模式，无额外依赖）..."
"$VENV_PIP" install --no-deps -e "$REPO_ROOT"

# ---- 3. 按需安装解析器 ----
echo "[3/4] 按需安装文档解析器..."
if [[ "$INSTALL_MINERU" == "true" ]]; then
  echo "  - 安装 MinerU（core）"
  "$VENV_PIP" install "mineru[core]"
else
  echo "  - 跳过 MinerU（如需安装，执行 INSTALL_MINERU=true bash reproduce/00_setup_env.sh）"
fi

if [[ "$INSTALL_DOCLING" == "true" ]]; then
  echo "  - 安装 Docling"
  "$VENV_PIP" install docling
else
  echo "  - 跳过 Docling（如需安装，执行 INSTALL_DOCLING=true bash reproduce/00_setup_env.sh）"
fi

# ---- 4. 验证安装 ----
echo "[4/4] 验证安装..."
"$VENV_PYTHON" - <<'EOF'
import sys
results = {}

try:
    import raganything
    results["raganything"] = f"OK (v{raganything.__version__})"
except Exception as e:
    results["raganything"] = f"FAIL: {e}"

try:
    import lightrag
    results["lightrag"] = "OK"
except Exception as e:
    results["lightrag"] = f"FAIL: {e}"

try:
    import dotenv
    results["python-dotenv"] = "OK"
except Exception as e:
    results["python-dotenv"] = f"FAIL: {e}"

try:
    import subprocess
    import pathlib
    venv_bin = pathlib.Path(sys.executable).resolve().parent
    mineru_cli = str(venv_bin / "mineru")
    r = subprocess.run([mineru_cli, "--version"], capture_output=True, text=True, timeout=10)
    results["mineru-cli"] = r.stdout.strip() or r.stderr.strip() or "OK"
except Exception as e:
    results["mineru-cli"] = f"FAIL: {e}"

try:
    import subprocess
    import pathlib
    venv_bin = pathlib.Path(sys.executable).resolve().parent
    docling_cli = str(venv_bin / "docling")
    r = subprocess.run([docling_cli, "--version"], capture_output=True, text=True, timeout=10)
    results["docling-cli"] = r.stdout.strip() or r.stderr.strip() or "OK"
except Exception as e:
    results["docling-cli"] = f"FAIL: {e}"

print("\n=== 安装验证结果 ===")
for k, v in results.items():
    status = "✓" if v.startswith("OK") else "✗"
    print(f"  {status} {k}: {v}")

failed = [k for k, v in results.items() if not v.startswith("OK")]
if failed:
    print(f"\n[警告] 以下组件安装失败: {failed}")
    print("可先使用最小环境完成处理器测试，再按需安装 MinerU 或 Docling。")
else:
    print("\n[成功] 所有组件安装完成！")
EOF

echo ""
echo "=== 下一步 ==="
echo "1. 编辑 reproduce/.env，填入你的 API Key"
echo "2. 运行: ./.venv/bin/python reproduce/01_run_pipeline.py path/to/your.pdf"
