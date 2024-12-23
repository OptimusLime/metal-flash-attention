import Metal
import QuartzCore

class LaplacianTester {
    func runTests(completion: @escaping (String) -> Void) {
        let problemSizes: [Int] = [
            7, 8, 9, 10,
            15, 16, 17, 18,
            23, 24, 25,
            31, 32, 33,
            47, 48, 49,
            63, 64, 65,
            103, 104, 112,
            126, 127, 128, 129,
            130, 131,
            135, 136, 137,
            143, 144, 145,
            151, 152, 153,
        ]
        
        let transposeStates: [(Bool, Bool)] = [
            (false, false),
            (false, true),
            (true, false),
        ]
        
        var totalTests = 0
        var completedTests = 0
        
        // Calculate total number of tests
        totalTests = problemSizes.count * transposeStates.count
        
        for problemSize in problemSizes {
            for transposeState in transposeStates {
                let n: UInt32 = .init(problemSize)
                let precision: GEMMOperandPrecision = .FP32
                
                // Set up the kernel
                var gemmDesc = GEMMDescriptor()
                gemmDesc.loadPreviousC = false
                gemmDesc.matrixDimensions = (M: n, N: n, K: n)
                gemmDesc.memoryPrecisions = (A: precision, B: precision, C: precision)
                gemmDesc.transposeState = transposeState
                
                // Test the kernel
                let result = profileProblemSize(descriptor: gemmDesc)
                completedTests += 1
                
                // Report progress
                let progress = "Test \(completedTests)/\(totalTests): Size \(problemSize), Transpose: \(transposeState), Result: \(result)"
                completion(progress)
            }
        }
        
        let finalResult = "Completed \(completedTests) Laplacian tests"
        completion(finalResult)
    }
    
    func runPerformanceTests(completion: @escaping (String) -> Void) {
        let transposeStates: [(Bool, Bool)] = [
            (false, false),
            (false, true),
            (true, false),
            (true, true),
        ]
        
        var results = ""
        results += "\nGEMM Performance Data:\n"
        
        for problemSize in 511...513 {
            for transposeState in transposeStates {
                let n: UInt32 = .init(problemSize)
                let precision: GEMMOperandPrecision = .BF16
                
                // Set up the kernel
                var gemmDesc = GEMMDescriptor()
                gemmDesc.loadPreviousC = false
                gemmDesc.matrixDimensions = (M: n, N: n, K: n)
                gemmDesc.memoryPrecisions = (A: precision, B: precision, C: precision)
                gemmDesc.transposeState = transposeState
                
                // Test the kernel
                let statistic = profileProblemSize(descriptor: gemmDesc)
                
                // Format results
                var result = String(format: "Size: %4d | ", problemSize)
                result += transposeState.0 ? "A^T " : "A   "
                result += transposeState.1 ? "B^T | " : "B   | "
                result += String(format: "%4d threads/core | %4d GFLOPS\n",
                                 statistic[1], statistic[0])
                
                results += result
                completion(results)  // Send incremental updates
            }
        }
        
        completion(results + "\nPerformance tests completed")
    }
    /// Returns:
    /// - lane 0: maximum achieved performance in GFLOPS
    /// - lane 1: occupancy in threads/core
    private func profileProblemSize(
        descriptor: GEMMDescriptor
    ) -> SIMD2<Int> {
        guard let matrixDimensions = descriptor.matrixDimensions,
              matrixDimensions.M == matrixDimensions.N,
              matrixDimensions.M == matrixDimensions.K else {
            fatalError("Matrix dimensions were invalid.")
        }
        let problemSize = Int(matrixDimensions.M)
        
        // Allocate FP32 memory for the operands
        var A = [Float](repeating: .zero, count: problemSize * problemSize)
        var B = [Float](repeating: .zero, count: problemSize * problemSize)
        var C = [Float](repeating: .zero, count: problemSize * problemSize)
        var previousOperandC = [Float](
            repeating: .zero, count: problemSize * problemSize)
        
        // Initialize A as the 2nd-order periodic Laplacian
        for diagonalID in 0..<problemSize {
            let diagonalAddress = diagonalID * problemSize + diagonalID
            A[diagonalAddress] = -2
            
            let leftColumnID = (diagonalID + problemSize - 1) % problemSize
            let leftSubDiagonalAddress = diagonalID * problemSize + leftColumnID
            A[leftSubDiagonalAddress] = 1
            
            let rightColumnID = (diagonalID + problemSize + 1) % problemSize
            let rightSubDiagonalAddress = diagonalID * problemSize + rightColumnID
            A[rightSubDiagonalAddress] = 1
        }
        
        // Initialize B to random numbers
        for rowID in 0..<problemSize {
            for columnID in 0..<problemSize {
                let address = rowID * problemSize + columnID
                let entry = Float.random(in: 0..<1)
                B[address] = entry
            }
        }
        
        // Initialize C to random numbers
        for rowID in 0..<problemSize {
            for columnID in 0..<problemSize {
                let address = rowID * problemSize + columnID
                let entry = Float.random(in: 0..<1)
                previousOperandC[address] = entry
            }
        }
        
        // Swap roles of matrices if testing left-hand side transposition
        if descriptor.transposeState!.A {
            swap(&A, &B)
        }
        
        // Multiply A with B
        var maxGFLOPS: Int = .zero
        var occupancy: Int = .zero
        do {
            // Generate the kernel
            GEMMKernel.register(descriptor: descriptor)
            let (kernel, pipeline) = GEMMKernel.pipelineCache[descriptor]!
            occupancy = pipeline.maxTotalThreadsPerThreadgroup
            
            // Create the buffers
            let bufferA = MTLContext.global
                .createBuffer(A, descriptor.memoryPrecisions!.A)
            let bufferB = MTLContext.global
                .createBuffer(B, descriptor.memoryPrecisions!.B)
            let bufferC = MTLContext.global
                .createBuffer(previousOperandC, descriptor.memoryPrecisions!.C)
            // Profile the latency of matrix multiplication
            for _ in 0..<(descriptor.loadPreviousC ? 1 : 15) {
                let duplicatedCommandCount: Int = (descriptor.loadPreviousC ? 1 : 20)
                
                // Encode the GPU command
                let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
                let encoder = commandBuffer.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipeline)
                encoder.setThreadgroupMemoryLength(
                    Int(kernel.threadgroupMemoryAllocation), index: 0)
                encoder.setBuffer(bufferA, offset: 0, index: 0)
                encoder.setBuffer(bufferB, offset: 0, index: 1)
                encoder.setBuffer(bufferC, offset: 0, index: 2)
                
                for _ in 0..<duplicatedCommandCount {
                    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
                        (target + Int(granularity) - 1) / Int(granularity)
                    }
                    let gridSize = MTLSize(
                        width: ceilDivide(problemSize, kernel.blockDimensions.N),
                        height: ceilDivide(problemSize, kernel.blockDimensions.M),
                        depth: 1)
                    let groupSize = MTLSize(
                        width: Int(kernel.threadgroupSize),
                        height: 1,
                        depth: 1)
                    encoder.dispatchThreadgroups(
                        gridSize, threadsPerThreadgroup: groupSize)
                }
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                
                // Determine the time taken
                let start = commandBuffer.gpuStartTime
                let end = commandBuffer.gpuEndTime
                let latency = end - start
                
                // Determine the amount of work done
                var operations = 2 * problemSize * problemSize * problemSize
                operations = operations * duplicatedCommandCount
                let gflops = Int(Double(operations) / Double(latency) / 1e9)
                
                // Update maximum GFLOPS
                maxGFLOPS = max(maxGFLOPS, gflops)
            }
            
            // Copy the results to C
            do {
                let precision = descriptor.memoryPrecisions!.C
                let raw = bufferC.contents()
                for rowID in 0..<problemSize {
                    for columnID in 0..<problemSize {
                        let address = rowID * problemSize + columnID
                        var entry32: Float
                        
                        switch precision {
                        case .FP32:
                            let casted = raw.assumingMemoryBound(to: Float.self)
                            entry32 = casted[address]
                        case .FP16:
                            let casted = raw.assumingMemoryBound(to: Float16.self)
                            let entry16 = casted[address]
                            entry32 = Float(entry16)
                        case .BF16:
                            let casted = raw.assumingMemoryBound(to: UInt16.self)
                            let entry16 = casted[address]
                            let entry16x2 = SIMD2<UInt16>(.zero, entry16)
                            entry32 = unsafeBitCast(entry16x2, to: Float.self)
                        }
                        C[address] = entry32
                    }
                }
            }
        }
        // Choose an error threshold
        func createErrorThreshold(precision: GEMMOperandPrecision) -> Float {
            switch precision {
            case .FP32: return 1e-5
            case .FP16: return 5e-3
            case .BF16: return 5e-2
            }
        }
        
        var errorThreshold: Float = 0
        do {
            let memoryPrecisions = descriptor.memoryPrecisions!
            let thresholdA = createErrorThreshold(precision: memoryPrecisions.A)
            let thresholdB = createErrorThreshold(precision: memoryPrecisions.B)
            let thresholdC = createErrorThreshold(precision: memoryPrecisions.C)
            errorThreshold = max(errorThreshold, thresholdA)
            errorThreshold = max(errorThreshold, thresholdB)
            errorThreshold = max(errorThreshold, thresholdC)
        }
        
        // Check the results
        var errorCount: Int = .zero
        for m in 0..<problemSize {
            for n in 0..<problemSize {
                // Find the source row IDs
                let leftRowID = (m + problemSize - 1) % problemSize
                let centerRowID = m
                let rightRowID = (m + problemSize + 1) % problemSize
                
                // Find the source scalars
                var leftSource: Float
                var centerSource: Float
                var rightSource: Float
                if descriptor.transposeState!.A {
                    leftSource = A[leftRowID * problemSize + n]
                    centerSource = A[centerRowID * problemSize + n]
                    rightSource = A[rightRowID * problemSize + n]
                } else if descriptor.transposeState!.B {
                    leftSource = B[n * problemSize + leftRowID]
                    centerSource = B[n * problemSize + centerRowID]
                    rightSource = B[n * problemSize + rightRowID]
                } else {
                    leftSource = B[leftRowID * problemSize + n]
                    centerSource = B[centerRowID * problemSize + n]
                    rightSource = B[rightRowID * problemSize + n]
                }
                
                // Find the expected result
                var expected = leftSource - 2 * centerSource + rightSource
                if descriptor.loadPreviousC {
                    if descriptor.transposeState!.A {
                        expected += previousOperandC[n * problemSize + m]
                    } else {
                        expected += previousOperandC[m * problemSize + n]
                    }
                }
                
                // Find the actual result
                var actual: Float
                if descriptor.transposeState!.A {
                    actual = C[n * problemSize + m]
                } else {
                    actual = C[m * problemSize + n]
                }
                
                // Report whether it is correct
                let error = (expected - actual).magnitude
                if error > errorThreshold {
                    if errorCount < 10 {
                        print("error: \(error) / ~1.000")
                        errorCount += 1
                    }
                }
            }
        }
        
        return SIMD2(maxGFLOPS, occupancy)
    }
}
