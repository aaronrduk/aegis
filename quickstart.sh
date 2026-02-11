#!/bin/bash

# SVAMITVA Feature Extraction - Quick Start Script

echo "🛰️ SVAMITVA Feature Extraction - Quick Start"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next Steps:"
echo "----------------------------------------"
echo "1. Prepare your data:"
echo "   - Place training images in: data/train/images/"
echo "   - Place training masks in: data/train/masks/"
echo "   - Place validation images in: data/val/images/"
echo "   - Place validation masks in: data/val/masks/"
echo ""
echo "   See DATA_PREPARATION.md for detailed instructions"
echo ""
echo "2. Train the model:"
echo "   python src/train.py"
echo ""
echo "3. Monitor training:"
echo "   tensorboard --logdir logs/"
echo ""
echo "4. Run the Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "5. Or process images in batch:"
echo "   python batch_process.py --checkpoint checkpoints/best_model.pth --input_dir data/test/images --output_dir outputs/"
echo ""
echo "6. Evaluate model performance:"
echo "   python evaluate.py --checkpoint checkpoints/best_model.pth --test_images data/test/images --test_masks data/test/masks"
echo ""
echo "----------------------------------------"
echo "For more information, see README.md"
echo ""
