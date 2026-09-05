import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Studio

// Tab 4 — inference playground: images, validation samples, webcam, video.
Item {
    id: page

    property int mode: 0     // 0 images · 1 validation · 2 webcam · 3 video
    property string videoPath: ""

    function cfg() {
        return {
            "conf": confSlider.value,
            "iou": iouSlider.value,
            "imgsz": parseInt(resCombo.currentText),
            "device": deviceValue(),
            "boxes": boxesCheck.checked,
            "labels": labelsCheck.checked,
            "conf_labels": confCheck.checked,
            "lineWidth": Math.round(widthSlider.value)
        }
    }
    function deviceValue() {
        if (!App.cudaAvailable) return "cpu"
        return devCombo.currentText === "CPU" ? "cpu" : "0"
    }
    function selectedModel() {
        return customWeights.text !== "" ? customWeights.text : modelCombo.currentText
    }

    Component.onCompleted: {
        modelCombo.currentIndex = App.defaultModelIndex
        if (App.models.length) Inference.loadModel(selectedModel())
    }

    FileDialog {
        id: ptDialog
        title: "Select custom .pt weights"
        nameFilters: ["PyTorch weights (*.pt)"]
        onAccepted: {
            customWeights.text = String(selectedFile).replace("file://", "")
            Inference.loadModel(customWeights.text)
        }
    }
    FileDialog {
        id: imgDialog
        title: "Select image(s)"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["Images (*.jpg *.jpeg *.png *.webp *.bmp)"]
        onAccepted: {
            var paths = []
            for (var i = 0; i < selectedFiles.length; ++i)
                paths.push(String(selectedFiles[i]))
            Inference.runImages(paths, page.cfg())
        }
    }
    FileDialog {
        id: videoDialog
        title: "Select a video"
        nameFilters: ["Video (*.mp4 *.avi *.mov)"]
        onAccepted: page.videoPath = String(selectedFile).replace("file://", "")
    }
    FileDialog {
        id: saveDialog
        property string source: ""
        title: "Save result as…"
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
                title: "🧪 Inference & Model Testing Playground"
                subtitle: "Test trained weights on images, validation samples, camera snapshots or video files."
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                spacing: 12

                // ---------------- settings ----------------
                ColumnLayout {
                    Layout.preferredWidth: page.width * 0.28
                    Layout.fillWidth: false
                    Layout.alignment: Qt.AlignTop
                    spacing: 12

                    Card {
                        SectionTitle { title: "1. Model & Inference Settings" }

                        Field {
                            label: "Discovered Checkpoints"
                            StudioCombo {
                                id: modelCombo
                                model: App.models
                                currentIndex: App.defaultModelIndex
                                onActivated: {
                                    customWeights.text = ""
                                    Inference.loadModel(currentText)
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            StudioButton {
                                text: "📁 Upload custom .pt"
                                Layout.fillWidth: true
                                onClicked: ptDialog.open()
                            }
                            StudioButton {
                                text: "🔄"
                                Layout.preferredWidth: 40
                                onClicked: App.refreshModels()
                            }
                        }
                        Text {
                            id: customWeights
                            visible: text !== ""
                            color: Theme.success
                            font.family: Theme.mono
                            font.pixelSize: Theme.fsTiny
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }

                        Field {
                            label: "Confidence Threshold — " + confSlider.value.toFixed(2)
                            StudioSlider { id: confSlider; from: 0.01; to: 1.0; value: 0.25; stepSize: 0.01 }
                        }
                        Field {
                            label: "NMS IoU Threshold — " + iouSlider.value.toFixed(2)
                            StudioSlider { id: iouSlider; from: 0.05; to: 1.0; value: 0.45; stepSize: 0.05 }
                        }
                        Field {
                            label: "Inference Resolution"
                            StudioCombo {
                                id: resCombo
                                model: ["320", "416", "512", "640", "768", "960", "1280"]
                                currentIndex: 3
                            }
                        }
                        Field {
                            label: "Compute Hardware"
                            StudioCombo {
                                id: devCombo
                                model: App.cudaAvailable ? ["Auto", "GPU (0)", "CPU"] : ["CPU"]
                            }
                        }
                    }

                    Card {
                        SectionTitle { title: "Visualization Settings" }
                        StudioCheck { id: boxesCheck; text: "Show Bounding Boxes"; checked: true }
                        StudioCheck { id: labelsCheck; text: "Show Labels"; checked: true }
                        StudioCheck { id: confCheck; text: "Show Confidence"; checked: true }
                        Field {
                            label: "Line Width — " + Math.round(widthSlider.value)
                            StudioSlider { id: widthSlider; from: 1; to: 10; value: 2; stepSize: 1 }
                        }
                    }
                }

                // ---------------- results ----------------
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                    spacing: 12

                    Card {
                        SectionTitle { title: "2. Input Mode & Detection Results" }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Repeater {
                                model: ["🖼️ Single/Multi Image", "📂 Validation Sample",
                                        "📷 Live Webcam", "🎥 Video File"]
                                delegate: StudioButton {
                                    required property var modelData
                                    required property int index
                                    text: modelData
                                    kind: page.mode === index ? "primary" : "secondary"
                                    Layout.fillWidth: true
                                    onClicked: page.mode = index
                                }
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            visible: !!Inference.modelInfo.name
                            implicitHeight: 34
                            radius: 8
                            color: Qt.rgba(0.06, 0.09, 0.16, 0.6)
                            border.color: Theme.border
                            border.width: 1
                            Text {
                                anchors.left: parent.left
                                anchors.leftMargin: 10
                                anchors.verticalCenter: parent.verticalCenter
                                text: "🧠 Model: " + (Inference.modelInfo.name || "—")
                                      + "  •  Task: " + (Inference.modelInfo.task || "—")
                                      + "  •  Classes: " + (Inference.modelInfo.classes || 0)
                                color: Theme.textDim
                                font.pixelSize: Theme.fsSmall
                            }
                        }
                        InfoNote {
                            text_: Inference.modelInfo.name ? "" :
                                   "Select or upload a valid .pt model checkpoint."
                            kind: "warn"
                        }
                        InfoNote { text_: Inference.busy; kind: "info" }

                        // --- per-mode controls ---
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            visible: page.mode === 0
                            StudioButton {
                                text: "🖼️ Choose image(s) & run inference"
                                kind: "primary"
                                Layout.fillWidth: true
                                enabled: !!Inference.modelInfo.name
                                onClicked: imgDialog.open()
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            visible: page.mode === 1
                            Field {
                                label: "Validation image"
                                StudioCombo {
                                    id: valCombo
                                    model: {
                                        var paths = App.datasetInfo.splitPaths || ({})
                                        var dir = paths["val"] || paths["valid"] || ""
                                        return dir ? Inference.imagesIn(dir) : []
                                    }
                                    displayText: currentText.split("/").pop()
                                }
                            }
                            StudioButton {
                                text: "▶ Run"
                                kind: "primary"
                                Layout.preferredWidth: 110
                                enabled: !!Inference.modelInfo.name && valCombo.count > 0
                                onClicked: Inference.runImages([valCombo.currentText], page.cfg())
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            visible: page.mode === 2
                            StudioButton {
                                text: "📷 Capture webcam frame & detect"
                                kind: "primary"
                                Layout.fillWidth: true
                                enabled: !!Inference.modelInfo.name
                                onClicked: Inference.captureWebcam(page.cfg())
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            visible: page.mode === 3
                            spacing: 8
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                StudioButton {
                                    text: "🎥 Choose video…"
                                    Layout.fillWidth: true
                                    onClicked: videoDialog.open()
                                }
                                StudioButton {
                                    text: "▶ Run Video Inference"
                                    kind: "primary"
                                    Layout.fillWidth: true
                                    enabled: page.videoPath !== "" && !!Inference.modelInfo.name
                                    onClicked: Inference.runVideo(page.videoPath, page.cfg())
                                }
                            }
                            Text {
                                text: page.videoPath
                                visible: page.videoPath !== ""
                                color: Theme.textMuted
                                font.family: Theme.mono
                                font.pixelSize: Theme.fsTiny
                                elide: Text.ElideMiddle
                                Layout.fillWidth: true
                            }
                            ProgressBarLine {
                                id: vidProgress
                                value: 0
                                caption: ""
                                visible: caption !== ""
                            }
                            Connections {
                                target: Inference
                                function onVideoProgress(pct, msg) {
                                    vidProgress.value = pct
                                    vidProgress.caption = msg
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                visible: Inference.videoOutput !== ""
                                spacing: 8
                                Text {
                                    text: "✅ " + Inference.videoOutput
                                    color: Theme.success
                                    font.pixelSize: Theme.fsSmall
                                    elide: Text.ElideMiddle
                                    Layout.fillWidth: true
                                }
                                StudioButton {
                                    text: "📥 Save annotated video…"
                                    onClicked: { saveDialog.source = Inference.videoOutput; saveDialog.open() }
                                }
                                StudioButton {
                                    text: "▶ Open"
                                    kind: "ghost"
                                    onClicked: App.revealPath(Inference.videoOutput)
                                }
                            }
                        }
                    }

                    // --- results ---
                    Repeater {
                        model: Inference.results
                        delegate: Card {
                            required property var modelData
                            Layout.fillWidth: true

                            RowLayout {
                                Layout.fillWidth: true
                                Text {
                                    text: modelData.name
                                    color: Theme.text
                                    font.pixelSize: Theme.fsBody
                                    font.bold: true
                                    Layout.fillWidth: true
                                    elide: Text.ElideMiddle
                                }
                                StatusBadge {
                                    text_: modelData.count + " detection(s)"
                                    tint: modelData.count > 0 ? Theme.success : Theme.textMuted
                                }
                                StatusBadge {
                                    text_: modelData.ms.toFixed(1) + " ms"
                                    tint: Theme.accent
                                }
                            }
                            Text {
                                text: "Latency — preprocess " + modelData.preprocess + " ms · inference "
                                      + modelData.inference + " ms · postprocess " + modelData.postprocess + " ms"
                                color: Theme.textMuted
                                font.pixelSize: Theme.fsTiny
                                Layout.fillWidth: true
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Repeater {
                                    model: [{ src: modelData.original, cap: "Original" },
                                            { src: modelData.annotated, cap: "Detections" }]
                                    delegate: ColumnLayout {
                                        required property var modelData
                                        Layout.fillWidth: true
                                        spacing: 4
                                        Rectangle {
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 300
                                            radius: 8
                                            color: "#05080f"
                                            border.color: Theme.border
                                            border.width: 1
                                            clip: true
                                            Image {
                                                anchors.fill: parent
                                                anchors.margins: 4
                                                fillMode: Image.PreserveAspectFit
                                                cache: false
                                                asynchronous: true
                                                source: "file://" + modelData.src
                                            }
                                        }
                                        Text {
                                            text: modelData.cap
                                            color: Theme.textMuted
                                            font.pixelSize: Theme.fsTiny
                                            Layout.fillWidth: true
                                            horizontalAlignment: Text.AlignHCenter
                                        }
                                    }
                                }
                            }

                            DataTable {
                                visible: modelData.detections.length > 0
                                columns: [{ title: "Class", key: "cls", width: 160 },
                                          { title: "Confidence", key: "conf", width: 110 },
                                          { title: "Coordinates [x1, y1, x2, y2]", key: "coords" }]
                                rows: modelData.detections
                                maxHeight: 220
                            }

                            StudioButton {
                                text: "📥 Save annotated image…"
                                Layout.fillWidth: true
                                onClicked: { saveDialog.source = modelData.annotated; saveDialog.open() }
                            }
                        }
                    }
                }
            }
            Item { Layout.preferredHeight: 10 }
        }
    }
}
