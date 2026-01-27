#!/bin/bash
# Test script for blur detection

echo "Testing blur detection with testimages..."
echo ""

# Run the app in test mode (you'll need to run the GUI manually, but this shows the approach)
# For now, we'll just verify the build worked
cargo build --release

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "To test blur detection:"
    echo "1. Run: ./target/release/imagestacker (or cargo run --release)"
    echo "2. Load images from ./testimages/"
    echo "3. Click 'Align Images'"
    echo "4. Check the terminal output for blur detection results"
    echo ""
    echo "You should see output like:"
    echo "  === BLUR DETECTION STARTING ==="
    echo "  Analyzing N images for sharpness..."
    echo "  [0] image.jpg: sharpness = 123.45"
    echo "  ..."
    echo "  === FILTERING SUMMARY ==="
else
    echo "✗ Build failed!"
    exit 1
fi
