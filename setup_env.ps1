Write-Host "Setting up Python virtual environment..."
$venvPath = "$PWD\venv"
python -m venv $venvPath
Write-Host "Activating virtual environment..."
& "$venvPath\Scripts\Activate.ps1"
Write-Host "Upgrading pip..."
pip install --upgrade pip
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
Write-Host "Installation complete."
