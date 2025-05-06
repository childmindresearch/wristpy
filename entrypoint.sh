#!/bin/bash
# Wristpy Docker Entrypoint Script

CMD="poetry run wristpy $INPUT_DIR --output $OUTPUT_DIR"


if [ -n "$OUTPUT_TYPE" ]; then
  CMD="$CMD --output-filetype $OUTPUT_TYPE"
fi

CMD="$CMD --calibrator $CALIBRATOR"
CMD="$CMD --activity-metric $ACTIVITY_METRIC"
CMD="$CMD --epoch-length $EPOCH_LENGTH"


if [ -n "$NONWEAR" ]; then
  IFS="," read -ra ALGOS <<< "$NONWEAR"
  for algo in "${ALGOS[@]}"; do
    CMD="$CMD --nonwear-algorithm $algo"
  done
fi

if [ -n "$THRESHOLDS" ]; then
  THRESHOLD_VALUES=$(echo $THRESHOLDS | tr "," " ")
  if [ -n "$THRESHOLD_VALUES" ]; then
    CMD="$CMD --thresholds $THRESHOLD_VALUES"
  fi
fi

echo "Running: $CMD"
exec $CMD