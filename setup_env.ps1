# Remove old environment if exists
if (Test-Path "deepvessel_env") {
    Remove-Item -Recurse -Force deepvessel_env
}

# Create new venv
python -m venv deepvessel_env

# Activate venv
.\deepvessel_env\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt
