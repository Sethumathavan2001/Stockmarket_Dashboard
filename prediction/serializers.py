from rest_framework import serializers


class PredictionHorizonSerializer(serializers.Serializer):
    horizon_minutes = serializers.IntegerField()
    predicted_return_percent = serializers.FloatField()
    predicted_open = serializers.FloatField()
    predicted_high = serializers.FloatField()
    predicted_low = serializers.FloatField()
    predicted_close = serializers.FloatField()
    direction = serializers.ChoiceField(choices=["UP", "DOWN", "FLAT"])
    confidence = serializers.FloatField()
    validation_mae = serializers.FloatField()
    validation_directional_accuracy = serializers.FloatField()


class CompanyPredictionSerializer(serializers.Serializer):
    company = serializers.CharField()
    timestamp = serializers.CharField()
    latest_candle_timestamp = serializers.CharField()
    current_price = serializers.FloatField()
    data_source = serializers.CharField()
    market_status = serializers.CharField()
    data_stale = serializers.BooleanField()
    historical_candles_used = serializers.IntegerField()
    model_version = serializers.CharField()
    model_trained_at = serializers.CharField()
    predictions = PredictionHorizonSerializer(many=True)
    disclaimer = serializers.CharField()
