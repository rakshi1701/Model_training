import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Studio

// Tab 5 — export studio: compile a checkpoint into a deployment runtime.
Item {
    id: page

    property var formatKeys: {
        var out = []
        var f = Exporter.formats
        for (var i = 0; i < f.length; ++i) out.push(f[i].key)
        return out
    }
    property string currentFormat: formatKeys.length ? formatKeys[fmtCombo.currentIndex] : "onnx"

    FileDialog {
        id: saveDialog
        property string source: ""
        title: "Save exported model as…"
        fileMode: FileDialog.SaveFile
        onAccepted: App.saveCopy(source, selectedFile)
    }

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: page.width - 28
            x: 14
            y: 12
            spacing: 12

            SectionTitle {
                title: "📦 Model Export & Production Deployment Studio"
                subtitle: "Convert trained weights into high-performance edge and cloud runtimes."
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                spacing: 12

                Card {
                    Layout.preferredWidth: page.width * 0.38
                    Layout.fillWidth: false
                    Layout.alignment: Qt.AlignTop
                    SectionTitle { title: "1. Export Configuration" }

                    Field {
                        label: "Model Checkpoint"
                        StudioCombo {
                            id: modelCombo
                            model: App.models
                            currentIndex: App.defaultModelIndex
                        }
                    }
                    StudioButton {
                        text: "🔄 Rescan checkpoints"
                        kind: "ghost"
                        Layout.fillWidth: true
                        onClicked: App.refreshModels()
                    }
                    Field {
                        label: "Target Runtime Format"
                        StudioCombo {
                            id: fmtCombo
                            model: {
                                var out = []
                                var f = Exporter.formats
                                for (var i = 0; i < f.length; ++i) out.push(f[i].label)
                                return out
                            }
                        }
                    }
                    Field {
                        label: "Input Resolution"
                        StudioCombo {
                            id: resCombo
                            model: ["320", "416", "512", "640", "768", "960", "1280"]
                            currentIndex: 3
                        }
                    }
                    StudioCheck { id: fp16Check; text: "FP16 Half-Precision (2x speed on GPU)" }
                    StudioCheck { id: dynCheck; text: "Dynamic Axes (Variable Batch & Size)" }
                    StudioCheck { id: simpCheck; text: "Simplify ONNX Graph"; checked: true }
                    StudioCheck { id: int8Check; text: "INT8 Quantization" }
                    Text {
                        text: "Compute: " + ((App.cudaAvailable && page.currentFormat === "engine") ? "GPU 0" : "CPU")
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsTiny
                        Layout.fillWidth: true
                    }
                }

                Card {
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                    SectionTitle { title: "2. Export Pipeline & Verification" }

                    InfoNote {
                        text_: "Target: " + page.currentFormat.toUpperCase()
                               + "  •  Checkpoint: " + App.basename(modelCombo.currentText)
                               + "  •  Size: " + resCombo.currentText + "px"
                        kind: "info"
                    }

                    StudioButton {
                        text: Exporter.busy !== "" ? "⏳ " + Exporter.busy : "🚀 Compile & Export Model"
                        kind: "primary"
                        enabled: Exporter.busy === "" && modelCombo.currentText !== ""
                        Layout.fillWidth: true
                        onClicked: Exporter.exportModel({
                            "model": modelCombo.currentText,
                            "format": page.currentFormat,
                            "imgsz": parseInt(resCombo.currentText),
                            "half": fp16Check.checked,
                            "dynamic": dynCheck.checked,
                            "simplify": simpCheck.checked,
                            "int8": int8Check.checked,
                            "device": (App.cudaAvailable && page.currentFormat === "engine") ? "0" : "cpu"
                        })
                    }

                    InfoNote {
                        text_: Exporter.result.ok === true
                               ? "🎉 Exported: " + Exporter.result.path
                                 + "\n📦 " + (Exporter.result.packaged ? "Packaged zip size: " : "Binary size: ")
                                 + Exporter.result.size
                               : ""
                        kind: "success"
                    }
                    InfoNote {
                        text_: Exporter.result.ok === false ? "Export error: " + Exporter.result.error : ""
                        kind: "error"
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        visible: Exporter.result.ok === true
                        spacing: 8
                        StudioButton {
                            text: "📥 Save exported file…"
                            Layout.fillWidth: true
                            onClicked: { saveDialog.source = Exporter.result.path; saveDialog.open() }
                        }
                        StudioButton {
                            text: "📂 Open folder"
                            kind: "ghost"
                            onClicked: App.revealPath(Exporter.result.path)
                        }
                    }

                    Text {
                        text: "Exports land in yolo_workspace/exports/ (directory formats such as "
                              + "OpenVINO are zipped automatically for 1-click portability)."
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsTiny
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }
            }
            Item { Layout.preferredHeight: 10 }
        }
    }
}
