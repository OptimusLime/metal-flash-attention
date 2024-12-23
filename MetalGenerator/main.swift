// MetalGenerator/main.swift
import Foundation

class MetalCodeGenerator {
    private let fileManager = FileManager.default
    
    func generateMetalFiles(destinationURL: URL) throws {
        print("Generating Metal files at: \(destinationURL.path)")
        
        // Create directories if they don't exist
        try fileManager.createDirectory(at: destinationURL,
                                      withIntermediateDirectories: true,
                                      attributes: nil)
        
        try generateAttentionMetal(destinationURL: destinationURL)
        try generateGEMMMetal(destinationURL: destinationURL)
        
        print("Successfully generated:")
        print("- \(destinationURL.appendingPathComponent("Attention.metal").path)")
        print("- \(destinationURL.appendingPathComponent("GEMM.metal").path)")
    }
    
   func generateAttentionMetal(destinationURL: URL) throws {
    // Create base descriptor that matches the original file's capabilities
    var descriptor = AttentionKernelDescriptor()
    
    // Matrix dimensions matching the function constants
    descriptor.headDimension = 64  // matches D in original
    descriptor.blockDimensions = (
        parallelization: 64,  // matches R_simd
        traversal: 64,        // matches C_simd
        head: 64              // matches D_simd = (D + 7) / 8 * 8
    )
    
    // Memory and register precisions
    descriptor.memoryPrecisions = [
        .Q: .FP16,
        .K: .FP16,
        .V: .FP16,
        .O: .FP16,
        .L: .FP32,
        .D: .FP32,
        .dO: .FP16,
        .dV: .FP16,
        .dK: .FP16,
        .dQ: .FP16
    ]
    
    descriptor.registerPrecisions = [
        .Q: .FP32,
        .K: .FP32,
        .V: .FP32,
        .O: .FP32,
        .L: .FP32,
        .D: .FP32,
        .dO: .FP32,
        .dV: .FP32,
        .dK: .FP32,
        .dQ: .FP32
    ]
    
    // Transpose states matching Q_trans, K_trans, etc.
    descriptor.transposeState = [
        .Q: false,
        .K: false,
        .V: false,
        .O: false,
        .dO: false,
        .dV: false,
        .dK: false,
        .dQ: false
    ]
    
    // Cache states for performance
    descriptor.cacheState = [
        .Q: true,
        .K: true,
        .V: true,
        .O: true,
        .dO: true,
        .dV: true,
        .dK: true,
        .dQ: true
    ]
    
    // Async preferences matching fuse_async_loads
    descriptor.preferAsyncCache = true
    descriptor.preferAsyncLoad = true
    
    // Create descriptors for each kernel type
    func createKernelDescriptor(type: AttentionKernelType) -> AttentionKernelDescriptor {
        var kernelDescriptor = descriptor
        kernelDescriptor.type = type
        
        // Adjust cache states based on kernel type
        switch type {
        case .forward:
            kernelDescriptor.cacheState = [
                .Q: true,
                .K: true,
                .V: true,
                .O: true
            ]
        case .backwardQuery:
            kernelDescriptor.cacheState = [
                .dO: true,
                .K: true,
                .V: true,
                .dQ: true
            ]
        case .backwardKeyValue:
            kernelDescriptor.cacheState = [
                .Q: true,
                .dO: true,
                .dK: true,
                .dV: true
            ]
        }
        
        return kernelDescriptor
    }
    
    // Create kernels for each type
    let forwardKernel = AttentionKernel(descriptor: createKernelDescriptor(type: .forward))
    let backwardQueryKernel = AttentionKernel(descriptor: createKernelDescriptor(type: .backwardQuery))
    let backwardKeyValueKernel = AttentionKernel(descriptor: createKernelDescriptor(type: .backwardKeyValue))
    
    // Get the source code
    let combinedSource = """
        //
        //  Attention.metal
        //  FlashAttention
        //
        
        #include <metal_stdlib>
        #include <simdgroup>
        
        // Function Constants - Dimensions
        constant uint R [[function_constant(0)]];  // row dimension (output sequence)
        constant uint C [[function_constant(1)]];  // column dimension (input sequence)
        constant uint H [[function_constant(2)]];  // head count
        constant uint D [[function_constant(3)]];  // head dimension
        
        // Matrix transpose states
        constant bool Q_trans [[function_constant(10)]];
        constant bool K_trans [[function_constant(11)]];
        constant bool V_trans [[function_constant(12)]];
        constant bool O_trans [[function_constant(13)]];
        
        using namespace metal;
        
        // Forward Pass Implementation
        \(forwardKernel.createSource())
        
        // Backward Query Implementation
        \(backwardQueryKernel.createSource())
        
        // Backward Key-Value Implementation
        \(backwardKeyValueKernel.createSource())
        """
        
        let attentionURL = destinationURL.appendingPathComponent("Attention.metal")
        try combinedSource.write(to: attentionURL, atomically: true, encoding: .utf8)
    }

    func generateGEMMMetal(destinationURL: URL) throws {
        // Create a base GEMM descriptor similar to the example
        var gemmDesc = GEMMDescriptor()
        gemmDesc.matrixDimensions = (M: 64, N: 64, K: 64)  // Example dimensions
        gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
        gemmDesc.transposeState = (A: false, B: false)
        
        // Create kernel descriptor
        var gemmKernelDesc = GEMMKernelDescriptor(descriptor: gemmDesc)
        gemmKernelDesc.leadingBlockDimensions = (A: 64, B: 64, C: 64)
        gemmKernelDesc.preferAsyncStore = true
        
        // Create kernel and get source
        let gemmKernel = GEMMKernel(descriptor: gemmKernelDesc)
        let source = gemmKernel.createSource()
        
        // Write to file
        let gemmURL = destinationURL.appendingPathComponent("GEMM.metal")
        try source.write(to: gemmURL, atomically: true, encoding: .utf8)
    }
}

// Main execution
let fileManager = FileManager.default

guard let projectDir = ProcessInfo.processInfo.environment["PROJECT_DIR"] else {
    print("Error: PROJECT_DIR environment variable not set")
    exit(1)
}

let sourceURL = URL(fileURLWithPath: projectDir)
    .appendingPathComponent("Shared")
    .appendingPathComponent("MetalSources")

do {
    try fileManager.createDirectory(at: sourceURL,
                                  withIntermediateDirectories: true)
    
    let generator = MetalCodeGenerator()
    try generator.generateMetalFiles(destinationURL: sourceURL)
    print("Successfully generated Metal files at: \(sourceURL.path)")
} catch {
    print("Error generating Metal files: \(error)")
    exit(1)
}
