package com.example.federatedlearning.tflite

import android.content.Context
import android.util.Log
import com.example.federatedlearning.data.PatientData
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

class TFLiteTrainer(private val context: Context) {
    private val TAG = "TFLiteTrainer"

    // Model parameters (mock neural network weights representing local TFLite transfer model)
    // 5 inputs -> 16 hidden nodes -> 8 hidden nodes -> 2 classes (healthy vs diabetic)
    private var weights_fc1 = Array(16) { FloatArray(5) { (Math.random() - 0.5).toFloat() } }
    private var weights_fc2 = Array(8) { FloatArray(16) { (Math.random() - 0.5).toFloat() } }
    private var weights_fc3 = Array(2) { FloatArray(8) { (Math.random() - 0.5).toFloat() } }

    fun loadModelWeights(parameters: Map<String, List<List<Double>>>) {
        try {
            // Unpack parameters if present
            parameters["fc1.weight"]?.let { weightsList ->
                for (i in 0 until 16) {
                    for (j in 0 until 5) {
                        weights_fc1[i][j] = weightsList[i][j].toFloat()
                    }
                }
            }
            parameters["fc2.weight"]?.let { weightsList ->
                for (i in 0 until 8) {
                    for (j in 0 until 16) {
                        weights_fc2[i][j] = weightsList[i][j].toFloat()
                    }
                }
            }
            parameters["fc3.weight"]?.let { weightsList ->
                for (i in 0 until 2) {
                    for (j in 0 until 8) {
                        weights_fc3[i][j] = weightsList[i][j].toFloat()
                    }
                }
            }
            Log.d(TAG, "Successfully loaded new global model weights.")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model weights: ${e.message}")
        }
    }

    fun exportModelWeights(): Map<String, List<List<Double>>> {
        val serialized = mutableMapOf<String, List<List<Double>>>()
        serialized["fc1.weight"] = weights_fc1.map { it.map { f -> f.toDouble() } }
        serialized["fc2.weight"] = weights_fc2.map { it.map { f -> f.toDouble() } }
        serialized["fc3.weight"] = weights_fc3.map { it.map { f -> f.toDouble() } }
        return serialized
    }

    /**
     * Executes local gradient descent training for 'epochs' over client dataset.
     */
    fun train(dataset: List<PatientData>, epochs: Int, logCallback: (String) -> Unit): Float {
        if (dataset.isEmpty()) {
            logCallback("No local data found. Skipping local training epoch steps.")
            return 0.0f
        }

        logCallback("Starting local gradient descent for $epochs epochs over ${dataset.size} diagnostic records...")
        
        var avgLoss = 0.0f
        val learningRate = 0.01f

        for (epoch in 1..epochs) {
            var totalLoss = 0.0f
            var correctCount = 0

            for (patient in dataset) {
                // Normalize features to reasonable bounds
                val x = floatArrayOf(
                    patient.glucose / 200.0f,
                    patient.bloodPressure / 120.0f,
                    patient.insulin / 300.0f,
                    patient.bmi / 50.0f,
                    patient.age / 80.0f
                )
                
                // Forward Pass
                // FC1
                val h1 = FloatArray(16)
                for (i in 0 until 16) {
                    var sum = 0.0f
                    for (j in 0 until 5) sum += x[j] * weights_fc1[i][j]
                    h1[i] = if (sum > 0) sum else 0.0f // ReLU
                }

                // FC2
                val h2 = FloatArray(8)
                for (i in 0 until 8) {
                    var sum = 0.0f
                    for (j in 0 until 16) sum += h1[j] * weights_fc2[i][j]
                    h2[i] = if (sum > 0) sum else 0.0f // ReLU
                }

                // FC3 (outputs)
                val out = FloatArray(2)
                for (i in 0 until 2) {
                    var sum = 0.0f
                    for (j in 0 until 8) sum += h2[j] * weights_fc3[i][j]
                    out[i] = sum
                }

                // Softmax
                val exp1 = exp(out[0].toDouble())
                val exp2 = exp(out[1].toDouble())
                val sumExp = exp1 + exp2
                val p0 = (exp1 / sumExp).toFloat()
                val p1 = (exp2 / sumExp).toFloat()

                // Calculate Cross Entropy Loss
                val target = patient.outcome
                val loss = if (target == 0) -Math.log(p0.toDouble()).toFloat() else -Math.log(p1.toDouble()).toFloat()
                totalLoss += loss

                // Prediction Check
                val pred = if (p1 > p0) 1 else 0
                if (pred == target) correctCount++

                // Backward Pass (Gradient Descent Updates)
                // Error gradients for outputs
                val grad_out = FloatArray(2)
                grad_out[0] = if (target == 0) p0 - 1.0f else p0
                grad_out[1] = if (target == 1) p1 - 1.0f else p1

                // Update weights_fc3
                for (i in 0 until 2) {
                    for (j in 0 until 8) {
                        weights_fc3[i][j] -= learningRate * grad_out[i] * h2[j]
                    }
                }

                // Backpropagate error to hidden layer 2
                val grad_h2 = FloatArray(8)
                for (j in 0 until 8) {
                    var sum = 0.0f
                    for (i in 0 until 2) sum += grad_out[i] * weights_fc3[i][j]
                    grad_h2[j] = if (h2[j] > 0) sum else 0.0f // ReLU gradient
                }

                // Update weights_fc2
                for (i in 0 until 8) {
                    for (j in 0 until 16) {
                        weights_fc2[i][j] -= learningRate * grad_h2[i] * h1[j]
                    }
                }
            }
            
            avgLoss = totalLoss / dataset.size
            val acc = correctCount.toFloat() / dataset.size
            logCallback("Epoch $epoch/$epochs completed. Average Loss: ${String.format("%.4f", avgLoss)}, Accuracy: ${String.format("%.2f", acc * 100)}%")
        }

        return avgLoss
    }
}
