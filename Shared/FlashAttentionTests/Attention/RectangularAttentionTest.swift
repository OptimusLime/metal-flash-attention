import Metal

class RectangularAttentionTester {
    // First function: Main test runner with UI feedback
    func runTests(completion: @escaping (String) -> Void) {
        var totalErrors = 0
        
        for testIndex in 0..<15 {
            var randomVecFloat = SIMD2<Float>.random(in: 0..<1)
            randomVecFloat = randomVecFloat * randomVecFloat * randomVecFloat
            var randomInts = SIMD2<Int>(randomVecFloat * SIMD2(128, 128))
            randomInts.replace(with: .one, where: randomInts .== .zero)
            
            var matrixDimensions = (
                row: UInt32(randomInts[0]),
                column: UInt32.zero,
                head: UInt16(randomInts[1]))
            if Float.random(in: 0..<1) < 0.5 {
                matrixDimensions.column = UInt32.random(in: 1...10)
            } else {
                matrixDimensions.column = UInt32.random(in: 10...128)
            }
            
            var descriptor = AttentionDescriptor()
            descriptor.lowPrecisionInputs = Bool.random()
            descriptor.lowPrecisionIntermediates = Bool.random()
            descriptor.matrixDimensions = matrixDimensions
            descriptor.transposeState = (
                Q: Bool.random(),
                K: Bool.random(),
                V: Bool.random(),
                O: Bool.random())
            
            let errors = runCorrectnessTest(descriptor: descriptor)
            totalErrors += errors
            
            let progress = "Test \(testIndex + 1)/15: \(errors) errors"
            completion(progress)
        }
        
        let finalResult = "Testing complete. Total errors: \(totalErrors)"
        completion(finalResult)
    }
    
    // Second function: The main test implementation
    private func runCorrectnessTest(descriptor: AttentionDescriptor) -> Int {
        guard let matrixDimensions = descriptor.matrixDimensions,
              let transposeState = descriptor.transposeState else {
            print("Error: Descriptor was incomplete.")
            return 1
        }
        
        var networkDesc = NetworkDescriptor()
        networkDesc.rowDimension = Int(matrixDimensions.row)
        networkDesc.columnDimension = Int(matrixDimensions.column)
        networkDesc.headDimension = Int(matrixDimensions.head)
        let network = Network(descriptor: networkDesc)
        
        // MARK: - Kernels
        let attentionDesc = descriptor
        
        func createKernel(type: AttentionKernelType) -> AttentionKernel {
            let attentionKernelDesc = attentionDesc.kernelDescriptor(type: type)
            let attentionKernel = AttentionKernel(descriptor: attentionKernelDesc)
            return attentionKernel
        }
        
        let kernelForward = createKernel(type: .forward)
        let kernelBackwardQuery = createKernel(type: .backwardQuery)
        let kernelBackwardKeyValue = createKernel(type: .backwardKeyValue)
        
        func createPipeline(kernel: AttentionKernel) -> MTLComputePipelineState {
            let device = MTLContext.global.device
            let source = kernel.createSource()
            let library = try! device.makeLibrary(source: source, options: nil)
            
            let functionConstants = MTLFunctionConstantValues()
            attentionDesc.setFunctionConstants(functionConstants)
            let function = try! library.makeFunction(
                name: "attention", constantValues: functionConstants)
            
            let pipelineDesc = MTLComputePipelineDescriptor()
            pipelineDesc.computeFunction = function
            pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
            return try! device.makeComputePipelineState(
                descriptor: pipelineDesc, options: [], reflection: nil)
        }
        
        let pipelineForward = createPipeline(kernel: kernelForward)
        let pipelineBackwardQuery = createPipeline(kernel: kernelBackwardQuery)
        let pipelineBackwardKeyValue = createPipeline(kernel: kernelBackwardKeyValue)
        // MARK: - Transpose Functions
        func transposeIn(_ input: [Float]) -> [Float] {
            let headDimension = Int(matrixDimensions.head)
            let sequenceDimension = input.count / headDimension
            
            var output = [Float](
                repeating: .zero, count: sequenceDimension * headDimension)
            for n in 0..<sequenceDimension {
                for d in 0..<headDimension {
                    let inputAddress = n * headDimension + d
                    let outputAddress = d * sequenceDimension + n
                    output[outputAddress] = input[inputAddress]
                }
            }
            return output
        }
        
        func transposeOut(_ output: [Float]) -> [Float] {
            let headDimension = Int(matrixDimensions.head)
            let sequenceDimension = output.count / headDimension
            
            var input = [Float](
                repeating: .zero, count: sequenceDimension * headDimension)
            for n in 0..<sequenceDimension {
                for d in 0..<headDimension {
                    let inputAddress = n * headDimension + d
                    let outputAddress = d * sequenceDimension + n
                    input[inputAddress] = output[outputAddress]
                }
            }
            return input
        }
        
        // Read the matrix inputs.
        var inputQ = network.Q
        var inputK = network.K
        var inputV = network.V
        var inputDerivativeO = network.dO
        if transposeState.Q {
            inputQ = transposeIn(inputQ)
        }
        if transposeState.K {
            inputK = transposeIn(inputK)
        }
        if transposeState.V {
            inputV = transposeIn(inputV)
        }
        if transposeState.O {
            inputDerivativeO = transposeIn(inputDerivativeO)
        }
        
        // MARK: - Buffer Functions
        func createArray(_ operand: AttentionOperand) -> [Float] {
            var size: Int
            switch operand {
            case .K, .V, .dV, .dK:
                size = Int(matrixDimensions.column) * Int(matrixDimensions.head)
            case .Q, .O, .dO, .dQ:
                size = Int(matrixDimensions.row) * Int(matrixDimensions.head)
            case .L, .D:
                size = Int(matrixDimensions.row)
            default:
                fatalError("Unsupported operand.")
            }
            return [Float](repeating: .zero, count: size)
        }
        
        func createBuffer(_ operand: AttentionOperand, contents: [Float]) -> MTLBuffer {
            let memoryPrecisions = attentionDesc.memoryPrecisions
            guard let precision = memoryPrecisions[operand] else {
                fatalError("Precision of operand \(operand) was not specified.")
            }
            return MTLContext.global.createBuffer(contents, precision)
        }
        
        // Create buffers
        let bufferQ = createBuffer(.Q, contents: inputQ)
        let bufferK = createBuffer(.K, contents: inputK)
        let bufferV = createBuffer(.V, contents: inputV)
        let bufferDerivativeO = createBuffer(.dO, contents: inputDerivativeO)
        
        // Allocate intermediates
        let bufferL = createBuffer(.L, contents: createArray(.L))
        let bufferD = createBuffer(.D, contents: createArray(.D))
        
        // Allocate outputs
        let bufferO = createBuffer(.O, contents: createArray(.O))
        let bufferDerivativeV = createBuffer(.dV, contents: createArray(.dV))
        let bufferDerivativeK = createBuffer(.dK, contents: createArray(.dK))
        let bufferDerivativeQ = createBuffer(.dQ, contents: createArray(.dQ))

                // MARK: - Command Buffer Execution
        @discardableResult
        func executeCommandBuffer(dispatchCount: Int) -> Double {
            let commandQueue = MTLContext.global.commandQueue
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            
            func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
                (target + Int(granularity) - 1) / Int(granularity)
            }
            
            func dispatch(
                kernel: AttentionKernel,
                pipeline: MTLComputePipelineState,
                along parallelizationDimension: Int
            ) {
                encoder.setComputePipelineState(pipeline)
                encoder.setThreadgroupMemoryLength(
                    Int(kernel.threadgroupMemoryAllocation), index: 0)
                
                let blockCount = ceilDivide(
                    parallelizationDimension, kernel.blockDimensions.parallelization)
                let gridSize = MTLSize(
                    width: blockCount,
                    height: 1,
                    depth: 1)
                let groupSize = MTLSize(
                    width: Int(kernel.threadgroupSize),
                    height: 1,
                    depth: 1)
                encoder.dispatchThreadgroups(
                    gridSize, threadsPerThreadgroup: groupSize)
            }
            
            encoder.setBuffer(bufferQ, offset: 0, index: 0)
            encoder.setBuffer(bufferK, offset: 0, index: 1)
            encoder.setBuffer(bufferV, offset: 0, index: 2)
            encoder.setBuffer(bufferO, offset: 0, index: 3)
            
            encoder.setBuffer(bufferL, offset: 0, index: 4)
            encoder.setBuffer(bufferD, offset: 0, index: 5)
            
            encoder.setBuffer(bufferDerivativeO, offset: 0, index: 6)
            encoder.setBuffer(bufferDerivativeV, offset: 0, index: 7)
            encoder.setBuffer(bufferDerivativeK, offset: 0, index: 8)
            encoder.setBuffer(bufferDerivativeQ, offset: 0, index: 9)
            
            for _ in 0..<dispatchCount {
                dispatch(
                    kernel: kernelForward,
                    pipeline: pipelineForward,
                    along: Int(matrixDimensions.row))
                dispatch(
                    kernel: kernelBackwardQuery,
                    pipeline: pipelineBackwardQuery,
                    along: Int(matrixDimensions.row))
                dispatch(
                    kernel: kernelBackwardKeyValue,
                    pipeline: pipelineBackwardKeyValue,
                    along: Int(matrixDimensions.column))
            }
            
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let start = commandBuffer.gpuStartTime
            let end = commandBuffer.gpuEndTime
            return end - start
        }
        
        // Execute the command buffer
        executeCommandBuffer(dispatchCount: 1)
        
        // MARK: - Results Collection
        func readBuffer(_ operand: AttentionOperand, contents: MTLBuffer) -> [Float] {
            let memoryPrecisions = attentionDesc.memoryPrecisions
            guard let precision = memoryPrecisions[operand] else {
                fatalError("Precision of operand \(operand) was not specified.")
            }
            
            var destination = createArray(operand)
            MTLContext.copy(contents, into: &destination, precision: precision)
            return destination
        }
        
        // Read intermediates
        var resultL = readBuffer(.L, contents: bufferL)
        var resultD = readBuffer(.D, contents: bufferD)
        for i in resultL.indices {
            resultL[i] /= 1.44269504089
        }
        for i in resultD.indices {
            resultD[i] /= 1 / Float(matrixDimensions.head).squareRoot()
        }
        
        // Read outputs
        var resultO = readBuffer(.O, contents: bufferO)
        var resultDerivativeV = readBuffer(.dV, contents: bufferDerivativeV)
        var resultDerivativeK = readBuffer(.dK, contents: bufferDerivativeK)
        var resultDerivativeQ = readBuffer(.dQ, contents: bufferDerivativeQ)
        
        // Handle transpose states
        if transposeState.Q {
            resultDerivativeQ = transposeOut(resultDerivativeQ)
        }
        if transposeState.K {
            resultDerivativeK = transposeOut(resultDerivativeK)
        }
        if transposeState.V {
            resultDerivativeV = transposeOut(resultDerivativeV)
        }
        if transposeState.O {
            resultO = transposeOut(resultO)
        }
                // MARK: - Validation
        // Get expected outputs from reference implementation
        let O = network.inferenceAttention()
        let L = (0..<Int(matrixDimensions.row)).map(network.createLTerm(rowID:))
        let D = (0..<Int(matrixDimensions.row)).map(network.createDTerm(rowID:))
        let dV = network.derivativeV()
        let dK = network.derivativeK()
        let dQ = network.derivativeQ()
        
        // Debug printing functions
        #if false
        func printVector(_ matrix: [Float]) {
            let sequenceDimension = matrix.count / 1
            
            for n in 0..<min(sequenceDimension, 10) {
                let matrixValue = matrix[n]
                var repr = String(format: "%.3f", matrixValue)
                while repr.count < 8 {
                    repr = " " + repr
                }
                print(repr, terminator: " ")
            }
            print()
        }
        
        func printMatrix(_ matrix: [Float]) {
            let sequenceDimension = matrix.count / Int(matrixDimensions.head)
            let headDimension = Int(matrixDimensions.head)
            
            for d in 0..<min(headDimension, 5) {
                for n in 0..<min(sequenceDimension, 10) {
                    let matrixAddress = n * headDimension + d
                    let matrixValue = matrix[matrixAddress]
                    var repr = String(format: "%.3f", matrixValue)
                    while repr.count < 8 {
                        repr = " " + repr
                    }
                    print(repr, terminator: " ")
                }
                print()
            }
        }
        
        print()
        print("Q:")
        printMatrix(network.Q)
        
        print()
        print("V:")
        printMatrix(network.V)
        
        print()
        print("O:")
        printMatrix(O)
        
        print()
        print("O:")
        printMatrix(resultO)
        
        print()
        print("L:")
        printVector(L)
        
        print()
        print("L:")
        printVector(resultL)
        
        print()
        print("D:")
        printVector(D)
        
        print()
        print("D:")
        printVector(resultD)
        
        print()
        print("dV:")
        printMatrix(dV)
        
        print()
        print("dV:")
        printMatrix(resultDerivativeV)
        
        print()
        print("dK:")
        printMatrix(dK)
        
        print()
        print("dK:")
        printMatrix(resultDerivativeK)
        
        print()
        print("dQ:")
        printMatrix(dQ)
        
        print()
        print("dQ:")
        printMatrix(resultDerivativeQ)
        #endif
        
        // Error checking
        var errorCount: Int = .zero
        func check(expected: [Float], actual: [Float], tolerance: Float) {
            guard expected.count == actual.count else {
                print("Error: Arrays had different length.")
                errorCount += 1
                return
            }
            
            for i in expected.indices {
                let error = (expected[i] - actual[i]).magnitude
                if error > tolerance || error.isNaN {
                    // Don't report errors in this case.
                    if (expected[i].isNaN || expected[i].isInfinite),
                       (actual[i].isNaN || actual[i].isInfinite ) {
                        continue
                    }
                    
                    // Update the error count in the outer scope.
                    if errorCount < 10 {
                        errorCount += 1
                        print("error: \(error) / ~1.000")
                        print("- expected[\(i)] = \(expected[i])")
                        print("-   actual[\(i)] = \(actual[i])")
                        print("- test configuration: \(descriptor)")
                    }
                }
            }
        }
        
        // Validate results with appropriate tolerances
        if descriptor.lowPrecisionInputs ||
            descriptor.lowPrecisionIntermediates {
            if matrixDimensions.column <= 20 {
                check(expected: O, actual: resultO, tolerance: 5e-2)
                check(expected: L, actual: resultL, tolerance: 1e-2)
                check(expected: D, actual: resultD, tolerance: 3e-1)
            } else {
                check(expected: O, actual: resultO, tolerance: 5e-2)
                check(expected: L, actual: resultL, tolerance: 7e-3)
                check(expected: D, actual: resultD, tolerance: 1e-1)
                check(expected: dV, actual: resultDerivativeV, tolerance: 5e-2)
                check(expected: dK, actual: resultDerivativeK, tolerance: 5e-2)
                check(expected: dQ, actual: resultDerivativeQ, tolerance: 5e-2)
            }
        } else {
            check(expected: O, actual: resultO, tolerance: 2e-5)
            check(expected: L, actual: resultL, tolerance: 2e-5)
            check(expected: D, actual: resultD, tolerance: 2e-5)
            check(expected: dV, actual: resultDerivativeV, tolerance: 2e-5)
            check(expected: dK, actual: resultDerivativeK, tolerance: 2e-5)
            check(expected: dQ, actual: resultDerivativeQ, tolerance: 2e-5)
        }
        
        return errorCount
    }
}
