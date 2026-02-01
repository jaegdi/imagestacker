#!/bin/bash
cd /home/dirk/devel/rust/imagestacker
echo "=== Testing stacking with channel normalization logs ==="
echo "Cleaning old bunches..."
rm -f testimages/bunches/*.png
rm -f testimages/final/*.png

echo "Running stacker..."
target/release/imagestacker --import testimages 2>&1 | tee test-stacking-output.log

echo ""
echo "=== Checking for channel conversion logs ==="
grep "Converting" test-stacking-output.log

echo ""
echo "=== Checking L0_B0004.png file size ==="
ls -lh testimages/bunches/L0_B0004.png

echo ""
echo "Done. Please check if L0_B0004.png is correct (not mostly black)."
