import SwiftUI

struct TestLog: Identifiable {
    let id = UUID()
    let timestamp: Date
    let message: String
    let isError: Bool
}

#if os(iOS)
typealias PlatformViewRepresentable = UIViewRepresentable
#elseif os(macOS)
typealias PlatformViewRepresentable = NSViewRepresentable
#endif

struct ContentView: View {
    var body: some View {
        #if os(iOS)
        MobileContentView()
        #elseif os(macOS)
        DesktopContentView()
        #endif
    }
}

#if os(iOS)
struct MobileContentView: View {
    @State private var testResults = "No tests run yet"
    @State private var isRunningTests = false
    @State private var selectedTest = 0
    @State private var testLogs: [TestLog] = []
    @State private var showPerformanceTests = false
    
    var body: some View {
        VStack {
            // Test Selection
            Picker("Select Test", selection: $selectedTest) {
                Text("Rectangular Attention").tag(0)
                Text("Adversarial Shape").tag(1)
                Text("Laplacian").tag(2)
                Text("Square Attention").tag(3)
            }
            .pickerStyle(.segmented)
            .padding()
            
            if selectedTest == 2 || selectedTest == 3 {
                Toggle("Run Performance Tests", isOn: $showPerformanceTests)
                    .padding()
            }
            
            Button(isRunningTests ? "Running Tests..." : "Run Selected Test") {
                runSelectedTest()
            }
            .disabled(isRunningTests)
            .padding()
            
            VStack(alignment: .leading) {
                Text("Test Logs")
                    .font(.headline)
                    .padding(.horizontal)
                
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(testLogs.reversed()) { log in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(log.timestamp, style: .time)
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                Text(log.message)
                                    .foregroundColor(log.isError ? .red : .primary)
                            }
                            .padding(.horizontal)
                            .padding(.vertical, 4)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.gray.opacity(0.1))
                            )
                            .padding(.horizontal)
                        }
                    }
                }
            }
            
            Button("Clear Logs") {
                testLogs.removeAll()
            }
            .padding()
            .disabled(testLogs.isEmpty)
        }
    }
    
    private func runSelectedTest() {
        isRunningTests = true
        testLogs.append(TestLog(
            timestamp: Date(),
            message: "Starting \(getTestName()) tests...",
            isError: false
        ))
        
        switch selectedTest {
        case 0:
            let tester = RectangularAttentionTester()
            tester.runTests { result in
                handleTestResult(result)
            }
        case 1:
            let tester = AdversarialShapeTester()
            tester.runTests { result in
                handleTestResult(result)
            }
        case 2:
            let tester = LaplacianTester()
            if showPerformanceTests {
                tester.runPerformanceTests { result in
                    handleTestResult(result)
                }
            } else {
                tester.runTests { result in
                    handleTestResult(result)
                }
            }
        case 3:
            let tester = SquareAttentionTester()
            if showPerformanceTests {
                tester.runPerformanceTests { result in
                    handleTestResult(result)
                }
            } else {
                tester.runTests { result in
                    handleTestResult(result)
                }
            }
        default:
            break
        }
    }
    
    private func getTestName() -> String {
        switch selectedTest {
        case 0: return "Rectangular Attention"
        case 1: return "Adversarial Shape"
        case 2: return showPerformanceTests ? "Laplacian Performance" : "Laplacian"
        case 3: return showPerformanceTests ? "Square Attention Performance" : "Square Attention"
        default: return "Unknown"
        }
    }
    
    private func handleTestResult(_ result: String) {
        let isError = result.contains("error") || result.contains("failed")
        testLogs.append(TestLog(
            timestamp: Date(),
            message: result,
            isError: isError
        ))
        
        if result.contains("complete") {
            isRunningTests = false
        }
    }
}
#endif

#if os(macOS)
struct DesktopContentView: View {
    @State private var testResults = "No tests run yet"
    @State private var isRunningTests = false
    @State private var selectedTest = 0
    @State private var testLogs: [TestLog] = []
    @State private var showPerformanceTests = false
    
    var body: some View {
        HSplitView {
            // Left side - Controls
            VStack(alignment: .leading, spacing: 20) {
                Picker("Select Test", selection: $selectedTest) {
                    Text("Rectangular Attention").tag(0)
                    Text("Adversarial Shape").tag(1)
                    Text("Laplacian").tag(2)
                    Text("Square Attention").tag(3)
                }
                .pickerStyle(.radioGroup)
                .padding()
                
                if selectedTest == 2 || selectedTest == 3 {
                    Toggle("Run Performance Tests", isOn: $showPerformanceTests)
                        .padding(.horizontal)
                }
                
                VStack(spacing: 10) {
                    Button(isRunningTests ? "Running Tests..." : "Run Selected Test") {
                        runSelectedTest()
                    }
                    .disabled(isRunningTests)
                    
                    Button("Clear Logs") {
                        testLogs.removeAll()
                    }
                    .disabled(testLogs.isEmpty)
                }
                .padding()
                
                Spacer()
            }
            .frame(minWidth: 200, maxWidth: 300)
            .padding()
            
            // Right side - Logs
            VStack {
                Text("Test Logs")
                    .font(.headline)
                    .padding()
                
                List(testLogs.reversed()) { log in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(log.timestamp, style: .time)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(log.message)
                            .foregroundColor(log.isError ? .red : .primary)
                    }
                    .padding(.vertical, 4)
                }
            }
            .frame(minWidth: 400)
        }
        .frame(minWidth: 700, minHeight: 400)
    }
    
    private func runSelectedTest() {
        isRunningTests = true
        testLogs.append(TestLog(
            timestamp: Date(),
            message: "Starting \(getTestName()) tests...",
            isError: false
        ))
        
        switch selectedTest {
        case 0:
            let tester = RectangularAttentionTester()
            tester.runTests { result in
                handleTestResult(result)
            }
        case 1:
            let tester = AdversarialShapeTester()
            tester.runTests { result in
                handleTestResult(result)
            }
        case 2:
            let tester = LaplacianTester()
            if showPerformanceTests {
                tester.runPerformanceTests { result in
                    handleTestResult(result)
                }
            } else {
                tester.runTests { result in
                    handleTestResult(result)
                }
            }
        case 3:
            let tester = SquareAttentionTester()
            if showPerformanceTests {
                tester.runPerformanceTests { result in
                    handleTestResult(result)
                }
            } else {
                tester.runTests { result in
                    handleTestResult(result)
                }
            }
        default:
            break
        }
    }
    
    private func getTestName() -> String {
        switch selectedTest {
        case 0: return "Rectangular Attention"
        case 1: return "Adversarial Shape"
        case 2: return showPerformanceTests ? "Laplacian Performance" : "Laplacian"
        case 3: return showPerformanceTests ? "Square Attention Performance" : "Square Attention"
        default: return "Unknown"
        }
    }
    
    private func handleTestResult(_ result: String) {
        let isError = result.contains("error") || result.contains("failed")
        testLogs.append(TestLog(
            timestamp: Date(),
            message: result,
            isError: isError
        ))
        
        if result.contains("complete") {
            isRunningTests = false
        }
    }
}
#endif

#Preview {
    ContentView()
}
