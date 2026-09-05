import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Studio

// Tab 1 — local training: hyperparameters, lifecycle controls, live metrics,
// the terminal stream, checkpoints and the host utilisation monitor.
Item {
    id: page

    property var preset: ({})

    function applyPreset(name) {
        preset = App.preset(name)
        familyCombo.currentIndex = Math.max(0, App.modelFamilies.indexOf(preset.model_family))
        page.syncSizes()
        var sizes = App.modelSizes(familyCombo.currentText)
        sizeCombo.currentIndex = Math.max(0, sizes.indexOf(preset.model_size))
        taskCombo.currentIndex = Math.max(0, taskCombo.model.indexOf(preset.task))
        epochsBox.value = preset.epochs || 100
        batchCombo.currentIndex = Math.max(0, batchCombo.model.indexOf(String(preset.batch)))
        imgszCombo.currentIndex = Math.max(0, imgszCombo.model.indexOf(String(preset.imgsz)))
        optCombo.currentIndex = Math.max(0, optCombo.model.indexOf(preset.optimizer))
        lrBox.setRealValue(preset.lr0 || 0.005)
        patienceBox.value = preset.patience || 50
        cosCheck.checked = !!preset.cos_lr
        wdBox.setRealValue(preset.weight_decay || 0.0005)
        freezeBox.value = preset.freeze || 0
    }

    function syncSizes() {
        sizeCombo.model = App.modelSizes(familyCombo.currentText)
    }

    function modelName() {
        var suffix = App.taskSuffix(taskCombo.currentText)
        return sizeCombo.currentText + suffix + (pretrainedCheck.checked ? ".pt" : ".yaml")
    }

    Component.onCompleted: {
        presetCombo.currentIndex = Math.min(1, App.presetNames.length - 1)
        applyPreset(presetCombo.currentText)
        runNameField.text = Train.runName
    }

    FileDialog {
        id: saveWeights
        property string source: ""
        title: "Save checkpoint as…"
        fileMode: FileDialog.SaveFile
        nameFilters: ["PyTorch weights (*.pt)"]
        onAccepted: App.saveCopy(source, selectedFile)
    }

    ScrollView {
        objectName: "trainScroll"
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: page.width - 28
            x: 14
            y: 12
            spacing: 12

            SectionTitle {
                title: "🏋️ Model Training Studio"
                subtitle: "Configure the run, launch it against the active dataset, and watch losses, mAP and hardware live."
            }

            // ---------- preset bar ----------
            Card {
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Field {
                        label: "🎯 Quick Training Preset Profile"
                        Layout.preferredWidth: 460
                        StudioCombo {
                            id: presetCombo
                            model: App.presetNames
                            onActivated: page.applyPreset(currentText)
                        }
                    }
                    Item { Layout.fillWidth: true }
                    StudioButton {
                        text: "✨ Apply Optuna Tuned Hyperparameters"
                        kind: "primary"
                        visible: App.hasTunedParams
                        onClicked: {
                            var t = App.tunedParams
                            if (t.lr0 !== undefined) lrBox.setRealValue(t.lr0)
                            if (t.weight_decay !== undefined) wdBox.setRealValue(t.weight_decay)
                            if (t.optimizer !== undefined) {
                                var i = optCombo.model.indexOf(String(t.optimizer))
                                if (i >= 0) optCombo.currentIndex = i
                            }
                            Notifier.notify("Tuned hyperparameters loaded into training settings.", "success")
                        }
                    }
                }
            }

            // ---------- hyperparameters ----------
            Card {
                SectionTitle { title: "⚙️ Hyperparameters & Hardware Configuration" }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 18
                    rowSpacing: 6

                    // 1. Architecture
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        spacing: 6
                        Text { text: "1. Architecture"; color: Theme.accent; font.bold: true; font.pixelSize: Theme.fsSmall }
                        Field {
                            label: "Family"
                            StudioCombo {
                                id: familyCombo
                                model: App.modelFamilies
                                onActivated: page.syncSizes()
                            }
                        }
                        Field {
                            label: "Model Size"
                            StudioCombo { id: sizeCombo; model: App.modelSizes("YOLO11") }
                        }
                        Field {
                            label: "Task"
                            StudioCombo {
                                id: taskCombo
                                model: ["detect", "segment", "classify", "pose", "obb"]
                            }
                        }
                        StudioCheck {
                            id: pretrainedCheck
                            text: "Start from Pretrained Weights"
                            checked: true
                        }
                        Text {
                            text: "→ " + page.modelName()
                            color: Theme.textMuted
                            font.family: Theme.mono
                            font.pixelSize: Theme.fsTiny
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                    }

                    // 2. Optimization
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        spacing: 6
                        Text { text: "2. Optimization"; color: Theme.accent; font.bold: true; font.pixelSize: Theme.fsSmall }
                        Field {
                            label: "Total Epochs"
                            StudioSpin {
                                id: epochsBox
                                realFrom: 1; realTo: 5000; value: 100
                                onValueChanged: Train.targetEpochs = value
                            }
                        }
                        Field {
                            label: "Batch Size"
                            hint: "-1 auto-fits batch to ~60% of GPU VRAM"
                            StudioCombo {
                                id: batchCombo
                                model: ["-1", "2", "4", "8", "16", "32", "64", "128"]
                                currentIndex: 4
                            }
                        }
                        Field {
                            label: "Image Size (px)"
                            StudioCombo {
                                id: imgszCombo
                                model: ["320", "416", "512", "640", "768", "960", "1280"]
                                currentIndex: 3
                            }
                        }
                        Field {
                            label: "Optimizer"
                            StudioCombo {
                                id: optCombo
                                model: ["auto", "AdamW", "SGD", "Adam", "NAdam", "RMSProp"]
                                currentIndex: 1
                            }
                        }
                    }

                    // 3. LR & decay
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        spacing: 6
                        Text { text: "3. Learning Rate & Decay"; color: Theme.accent; font.bold: true; font.pixelSize: Theme.fsSmall }
                        Field {
                            label: "Initial LR (lr0)"
                            StudioSpin {
                                id: lrBox
                                decimals: 5; realFrom: 0.00001; realTo: 1.0; realStep: 0.001
                                value: 500
                            }
                        }
                        Field {
                            label: "Early Stopping (epochs)"
                            StudioSpin { id: patienceBox; realFrom: 0; realTo: 500; value: 50 }
                        }
                        StudioCheck { id: cosCheck; text: "Cosine LR Schedule"; checked: true }
                        Field {
                            label: "Image Caching"
                            StudioCombo {
                                id: cacheCombo
                                model: ["False", "ram", "disk"]
                            }
                        }
                    }

                    // 4. Compute & checkpoints
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        spacing: 6
                        Text { text: "4. Compute & Checkpoints"; color: Theme.accent; font.bold: true; font.pixelSize: Theme.fsSmall }
                        Field {
                            label: "Target Compute"
                            StudioCombo { id: deviceCombo; model: App.deviceOptions }
                        }
                        Field {
                            label: "Dataloader Workers"
                            StudioSpin { id: workersBox; realFrom: 0; realTo: 32; value: 8 }
                        }
                        Field {
                            label: "Experiment Run Name"
                            StudioText {
                                id: runNameField
                                text: Train.runName
                                onEditingFinished: Train.runName = text
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Field {
                                label: "Weight Decay"
                                StudioSpin {
                                    id: wdBox
                                    decimals: 5; realFrom: 0; realTo: 0.1; realStep: 0.0005
                                    value: 50
                                }
                            }
                            Field {
                                label: "Freeze Layers"
                                StudioSpin { id: freezeBox; realFrom: 0; realTo: 50; value: 0 }
                            }
                        }
                        StudioCheck { id: resumeCheck; text: "Resume Checkpoint" }
                    }
                }
            }

            // ---------- action bar ----------
            Card {
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    StudioButton {
                        text: "🚀 Start Training Run"
                        kind: "primary"
                        Layout.preferredWidth: 220
                        enabled: App.hasDataset && !Train.active
                        onClicked: Train.start({
                            "model": page.modelName(),
                            "data": App.dataYamlPath,
                            "name": runNameField.text,
                            "epochs": epochsBox.value,
                            "batch": parseInt(batchCombo.currentText),
                            "imgsz": parseInt(imgszCombo.currentText),
                            "optimizer": optCombo.currentText,
                            "lr0": lrBox.realValue,
                            "patience": patienceBox.value,
                            "device": App.deviceValues[deviceCombo.currentIndex],
                            "workers": workersBox.value,
                            "resume": resumeCheck.checked,
                            "cache": cacheCombo.currentText,
                            "cos_lr": cosCheck.checked,
                            "weight_decay": wdBox.realValue,
                            "freeze": freezeBox.value
                        })
                    }
                    StudioButton {
                        text: "⏸ Pause"
                        enabled: Train.active && !Train.paused
                        onClicked: Train.pause()
                    }
                    StudioButton {
                        text: "▶ Resume"
                        enabled: Train.active && Train.paused
                        onClicked: Train.resume()
                    }
                    StudioButton {
                        text: "🛑 Terminate"
                        kind: "danger"
                        enabled: Train.active
                        onClicked: Train.terminate()
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        visible: Train.active
                        text: "⏱ " + Train.elapsedText
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsSmall
                    }
                    StatusBadge {
                        text_: Train.statusText
                        tint: Train.statusKind === "running" ? Theme.success
                            : Train.statusKind === "paused" ? Theme.warn : Theme.textMuted
                    }
                }
            }

            // ---------- live metrics ----------
            InfoNote {
                text_: Object.keys(Train.metrics).length === 0
                       ? "💡 Real-time metric curves and KPI badges render here as soon as the first epoch finishes."
                       : ""
                kind: "info"
            }

            ColumnLayout {
                visible: Object.keys(Train.metrics).length > 0
                Layout.fillWidth: true
                spacing: 12

                Card {
                    ProgressBarLine {
                        value: Train.metrics.progress || 0
                        caption: "Epoch Progress: " + (Train.metrics.epoch || 0) + "/"
                                 + (Train.metrics.totalEpochs || 0) + " ("
                                 + Math.round((Train.metrics.progress || 0) * 100)
                                 + "%) • Estimated ETA: " + (Train.metrics.eta || "—")
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    MetricCard {
                        label: "Epoch"
                        value: (Train.metrics.epoch || 0) + " / " + (Train.metrics.totalEpochs || 0)
                    }
                    MetricCard {
                        label: "Box Loss"
                        inverse: true
                        value: Train.metrics.boxLoss !== null && Train.metrics.boxLoss !== undefined
                               ? Train.metrics.boxLoss.toFixed(4) : "N/A"
                        delta: Train.metrics.boxDelta ? Train.metrics.boxDelta.toFixed(4) : ""
                    }
                    MetricCard {
                        label: "Class Loss"
                        inverse: true
                        value: Train.metrics.clsLoss !== null && Train.metrics.clsLoss !== undefined
                               ? Train.metrics.clsLoss.toFixed(4) : "N/A"
                        delta: Train.metrics.clsDelta ? Train.metrics.clsDelta.toFixed(4) : ""
                    }
                    MetricCard {
                        label: "mAP@50"
                        valueColor: Theme.success
                        value: Train.metrics.map50 !== null && Train.metrics.map50 !== undefined
                               ? Train.metrics.map50.toFixed(4) : "N/A"
                        delta: Train.metrics.map50Delta
                               ? (Train.metrics.map50Delta > 0 ? "+" : "") + Train.metrics.map50Delta.toFixed(4) : ""
                    }
                    MetricCard {
                        label: "mAP@50-95"
                        valueColor: Theme.success
                        value: Train.metrics.map5095 !== null && Train.metrics.map5095 !== undefined
                               ? Train.metrics.map5095.toFixed(4) : "N/A"
                        delta: Train.metrics.map5095Delta
                               ? (Train.metrics.map5095Delta > 0 ? "+" : "") + Train.metrics.map5095Delta.toFixed(4) : ""
                    }
                    MetricCard {
                        label: "Prec / Recall"
                        value: (Train.metrics.precision || 0).toFixed(2) + " / "
                               + (Train.metrics.recall || 0).toFixed(2)
                    }
                }

                Card {
                    TabBar {
                        id: chartTabs
                        Layout.fillWidth: true
                        background: Rectangle { color: "transparent" }
                        Repeater {
                            model: ["📉 Training & Validation Losses",
                                    "🎯 mAP & Accuracy Dynamics",
                                    "⚡ Learning Rate Schedule"]
                            delegate: TabButton {
                                required property var modelData
                                required property int index
                                text: modelData
                                implicitHeight: 32
                                font.pixelSize: Theme.fsSmall
                                background: Rectangle {
                                    color: chartTabs.currentIndex === index ? Theme.cardHover : "transparent"
                                    radius: 6
                                }
                                contentItem: Text {
                                    text: parent.text
                                    font: parent.font
                                    color: chartTabs.currentIndex === index ? Theme.text : Theme.textMuted
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    elide: Text.ElideRight
                                }
                            }
                        }
                    }
                    StackLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 310
                        currentIndex: chartTabs.currentIndex
                        MetricChart { seriesData: Train.lossSeries; yTitle: "Loss" }
                        MetricChart { seriesData: Train.accSeries; yTitle: "Score" }
                        MetricChart { seriesData: Train.lrSeries; yTitle: "LR" }
                    }
                }
            }

            // ---------- console + artifacts ----------
            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Card {
                    Layout.preferredWidth: page.width * 0.6
                    Layout.fillWidth: false
                    Layout.alignment: Qt.AlignTop
                    SectionTitle { title: "🖥️ Live Terminal Log Stream" }
                    LogView {
                        text_: Train.logText
                        implicitHeight: 300
                    }
                }

                Card {
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                    SectionTitle { title: "🏆 Checkpoints & Artifacts" }

                    InfoNote {
                        text_: Train.artifacts.best
                               ? "✨ best.pt ready (" + Train.artifacts.bestSize + ")" : ""
                        kind: "success"
                    }
                    StudioButton {
                        text: "📥 Save best.pt …"
                        visible: !!Train.artifacts.best
                        Layout.fillWidth: true
                        onClicked: { saveWeights.source = Train.artifacts.best; saveWeights.open() }
                    }
                    StudioButton {
                        text: "📥 Save last.pt …"
                        visible: !!Train.artifacts.last
                        Layout.fillWidth: true
                        onClicked: { saveWeights.source = Train.artifacts.last; saveWeights.open() }
                    }
                    StudioButton {
                        text: "📂 Open run folder"
                        kind: "ghost"
                        visible: !!Train.artifacts.dir
                        Layout.fillWidth: true
                        onClicked: App.revealPath(Train.artifacts.dir)
                    }
                    Text {
                        visible: !Train.artifacts.dir
                        text: "Training checkpoints and summary plots appear here."
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsSmall
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                    Image {
                        visible: !!Train.artifacts.plot
                        source: Train.artifacts.plot
                                ? "file://" + Train.artifacts.plot + "?t=" + (Train.artifacts.plotStamp || 0)
                                : ""
                        fillMode: Image.PreserveAspectFit
                        Layout.fillWidth: true
                        Layout.preferredHeight: 220
                        smooth: true
                    }
                }
            }

            // ---------- system utilisation ----------
            SystemMonitorPanel { }

            Item { Layout.preferredHeight: 10 }
        }
    }
}
