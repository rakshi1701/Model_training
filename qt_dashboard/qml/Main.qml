import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Studio
import "pages"

ApplicationWindow {
    id: win
    width: 1640
    height: 1000
    minimumWidth: 1180
    minimumHeight: 720
    visible: true
    title: "YOLO Studio — Training, Inference, Export & Tuning"
    color: Theme.bg

    // ---- toast plumbing --------------------------------------------------
    function notify(msg, kind) { toast.show(msg, kind) }
    Connections { target: Notifier; function onNotified(m, k) { toast.show(m, k) } }

    Connections { target: App;       function onToast(m, k) { win.notify(m, k) } }
    Connections { target: Train;     function onToast(m, k) { win.notify(m, k) } }
    Connections { target: Inference; function onToast(m, k) { win.notify(m, k) } }
    Connections { target: Exporter;  function onToast(m, k) { win.notify(m, k) } }
    Connections { target: Tuner;     function onToast(m, k) { win.notify(m, k) } }
    Connections {
        target: Kaggle
        enabled: Kaggle !== null
        function onToast(m, k) { win.notify(m, k) }
        function onStopInstructions(msg, url) {
            stopDialog.body = msg
            stopDialog.url = url
            stopDialog.open()
        }
    }

    // ---- shared file dialogs --------------------------------------------
    FileDialog {
        id: zipDialog
        title: "Select a YOLO dataset .zip"
        nameFilters: ["Zip archives (*.zip)"]
        onAccepted: App.importDatasetZip(selectedFile)
    }
    FileDialog {
        id: yamlDialog
        title: "Select a data.yaml"
        nameFilters: ["Dataset config (*.yaml *.yml)"]
        onAccepted: App.selectDatasetPath(selectedFile)
    }

    Dialog {
        id: stopDialog
        property string body: ""
        property string url: ""
        anchors.centerIn: parent
        width: 560
        modal: true
        title: "Stopping a Kaggle job"
        standardButtons: Dialog.Close
        background: Rectangle {
            color: "#0d1526"; radius: 12
            border.color: Theme.border; border.width: 1
        }
        header: Text {
            text: "  🛑 Stopping a Kaggle job"
            color: Theme.text; font.bold: true; font.pixelSize: Theme.fsTitle
            padding: 14
        }
        contentItem: ColumnLayout {
            spacing: 10
            Text {
                text: stopDialog.body
                color: Theme.textDim
                font.pixelSize: Theme.fsBody
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
            StudioButton {
                text: "🔗 Open the kernel page on Kaggle"
                onClicked: App.openUrl(stopDialog.url)
            }
        }
    }

    // ---- layout ----------------------------------------------------------
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ============================ SIDEBAR ============================
        Rectangle {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            color: Theme.bgAlt
            border.color: Theme.border
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12

                Text {
                    text: "⚡ YOLO Control Center"
                    color: Theme.text
                    font.pixelSize: 16
                    font.bold: true
                }

                // Hardware badge
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: hw.implicitHeight + 18
                    radius: 8
                    color: App.cudaAvailable ? Qt.rgba(0.06, 0.72, 0.51, 0.10)
                                             : Qt.rgba(0.96, 0.62, 0.04, 0.10)
                    border.width: 1
                    border.color: App.cudaAvailable ? Qt.rgba(0.06, 0.72, 0.51, 0.32)
                                                    : Qt.rgba(0.96, 0.62, 0.04, 0.32)
                    ColumnLayout {
                        id: hw
                        anchors.fill: parent
                        anchors.margins: 9
                        spacing: 2
                        Text {
                            text: App.cudaAvailable ? "⚡ NVIDIA CUDA Accelerated"
                                                    : "⚠️ CPU Compute Mode"
                            color: App.cudaAvailable ? Theme.green : Theme.warn
                            font.pixelSize: Theme.fsBody
                            font.bold: true
                        }
                        Text {
                            text: App.cudaAvailable
                                  ? App.gpuCount + "x GPU Device(s) Online"
                                  : "Training will use host CPU cores"
                            color: Theme.textMuted
                            font.pixelSize: Theme.fsSmall
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Text {
                            visible: App.cudaAvailable
                            text: App.gpuNames.join(", ")
                            color: Theme.textMuted
                            font.pixelSize: Theme.fsTiny
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                Text {
                    text: "📁 Active Dataset"
                    color: Theme.textDim
                    font.pixelSize: Theme.fsBody
                    font.bold: true
                }

                StudioCombo {
                    id: dsCombo
                    model: App.datasetNames
                    onActivated: App.selectDataset(App.datasetNames[currentIndex])

                    // Picking an item breaks a currentIndex binding for good, so
                    // follow the backend explicitly instead.
                    function sync() {
                        var i = App.datasetNames.indexOf(App.activeDatasetKey)
                        if (i >= 0 && i !== currentIndex)
                            currentIndex = i
                    }
                    Component.onCompleted: sync()
                    Connections {
                        target: App
                        function onActiveDatasetChanged() { dsCombo.sync() }
                        function onDatasetsChanged() { dsCombo.sync() }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    StudioButton {
                        text: "➕ Upload .zip"
                        Layout.fillWidth: true
                        onClicked: zipDialog.open()
                    }
                    StudioButton {
                        text: "🔍 Path…"
                        Layout.fillWidth: true
                        onClicked: yamlDialog.open()
                    }
                }
                StudioButton {
                    text: "🔄 Rescan workspace"
                    kind: "ghost"
                    Layout.fillWidth: true
                    onClicked: { App.refreshDatasets(); App.refreshModels() }
                }

                Text {
                    text: "Datasets are kept side by side — a new upload is added, not "
                          + "replacing what is already there."
                    color: Theme.textMuted
                    font.pixelSize: Theme.fsTiny
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }

                InfoNote {
                    text_: App.busy
                    kind: "info"
                }

                // Active dataset stats
                Rectangle {
                    Layout.fillWidth: true
                    visible: App.hasDataset
                    implicitHeight: dsStats.implicitHeight + 20
                    radius: 8
                    color: Qt.rgba(0.06, 0.09, 0.16, 0.75)
                    border.color: Theme.border
                    border.width: 1
                    ColumnLayout {
                        id: dsStats
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 4
                        Text {
                            text: "📄 " + (App.datasetInfo.name || "")
                            color: Theme.accent
                            font.pixelSize: Theme.fsBody
                            font.bold: true
                            Layout.fillWidth: true
                            elide: Text.ElideMiddle
                        }
                        Text {
                            text: {
                                var c = App.datasetInfo.counts || []
                                if (!c.length) return "🖼️ Images: none detected"
                                var parts = []
                                for (var i = 0; i < c.length; ++i)
                                    parts.push(c[i].split + ": " + c[i].count)
                                return "🖼️ Images — " + parts.join(" · ")
                            }
                            color: Theme.textDim
                            font.pixelSize: Theme.fsSmall
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                        Text {
                            text: {
                                var cl = App.datasetInfo.classes || []
                                var names = []
                                for (var i = 0; i < Math.min(5, cl.length); ++i)
                                    names.push(cl[i].name)
                                return "🏷️ Classes (" + cl.length + "): "
                                       + (names.join(", ") || "None")
                                       + (cl.length > 5 ? "…" : "")
                            }
                            color: Theme.textDim
                            font.pixelSize: Theme.fsSmall
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                        Text {
                            text: App.datasetInfo.yaml || ""
                            color: Theme.textMuted
                            font.pixelSize: Theme.fsTiny
                            font.family: Theme.mono
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                    }
                }
                InfoNote {
                    text_: App.hasDataset ? "" : "Select or upload a dataset to begin training."
                    kind: "info"
                }

                // Removing a dataset is only offered for folders inside
                // yolo_workspace/dataset/ — a custom path is somebody else's data.
                ColumnLayout {
                    Layout.fillWidth: true
                    visible: App.hasDataset && App.datasetInfo.removable === true
                    spacing: 6
                    property bool confirming: false

                    StudioButton {
                        text: "🗑 Remove “" + (App.datasetInfo.name || "") + "”"
                        visible: !parent.confirming
                        Layout.fillWidth: true
                        onClicked: parent.confirming = true
                    }
                    InfoNote {
                        visible: parent.confirming
                        text_: "Delete " + (App.datasetInfo.name || "") + " and everything in it?"
                        kind: "warn"
                    }
                    RowLayout {
                        visible: parent.confirming
                        Layout.fillWidth: true
                        spacing: 6
                        StudioButton {
                            text: "🗑 Delete"
                            kind: "danger"
                            Layout.fillWidth: true
                            onClicked: {
                                parent.parent.confirming = false
                                App.removeActiveDataset()
                            }
                        }
                        StudioButton {
                            text: "Cancel"
                            Layout.fillWidth: true
                            onClicked: parent.parent.confirming = false
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: Theme.border }
                Text {
                    text: "🚀 YOLO Vision Studio · Qt/QML edition"
                    color: Theme.textMuted
                    font.pixelSize: Theme.fsTiny
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                }
            }
        }

        // ============================ MAIN ==============================
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // Pipeline tracker
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 58
                color: Theme.bgAlt
                border.color: Theme.border
                border.width: 1
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 16
                    anchors.rightMargin: 16
                    spacing: 8

                    Repeater {
                        model: [
                            { t: "📁 1. Dataset",        b: App.hasDataset ? "Ready ✅" : "Waiting 📁", a: App.hasDataset },
                            { t: "⚙️ 2. Model & Tuning", b: "Configured ⚡", a: true },
                            { t: "🏋️ 3. Training",       b: Train.active ? (Train.paused ? "Paused ⏸" : "Active 🟢") : "Idle ⚪", a: Train.active },
                            { t: "🧪 4. Inference",      b: "Interactive", a: true },
                            { t: "📦 5. Export",         b: "Multi-Format", a: true }
                        ]
                        delegate: RowLayout {
                            required property var modelData
                            required property int index
                            spacing: 8
                            Rectangle {
                                implicitHeight: 34
                                implicitWidth: step.implicitWidth + 22
                                radius: 17
                                color: modelData.a ? Qt.rgba(0.22, 0.74, 0.97, 0.12) : Theme.card
                                border.width: 1
                                border.color: modelData.a ? Theme.borderStrong : Theme.border
                                RowLayout {
                                    id: step
                                    anchors.centerIn: parent
                                    spacing: 8
                                    Text {
                                        text: modelData.t
                                        color: modelData.a ? Theme.text : Theme.textMuted
                                        font.pixelSize: Theme.fsSmall
                                        font.bold: true
                                    }
                                    Rectangle {
                                        implicitHeight: 18
                                        implicitWidth: badge.implicitWidth + 14
                                        radius: 9
                                        color: Qt.rgba(1, 1, 1, 0.07)
                                        Text {
                                            id: badge
                                            anchors.centerIn: parent
                                            text: modelData.b
                                            color: Theme.textDim
                                            font.pixelSize: Theme.fsTiny
                                        }
                                    }
                                }
                            }
                            Text {
                                visible: index < 4
                                text: "➔"
                                color: Theme.textMuted
                                font.pixelSize: Theme.fsBody
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    StatusBadge {
                        text_: Train.statusText
                        tint: Train.statusKind === "running" ? Theme.success
                            : Train.statusKind === "paused" ? Theme.warn : Theme.textMuted
                    }
                }
            }

            // Tabs
            TabBar {
                id: tabs
                objectName: "mainTabs"
                Layout.fillWidth: true
                background: Rectangle { color: Theme.bg }
                Repeater {
                    model: ["🏋️ Local Training & Metrics",
                            "☁️ Kaggle Cloud GPU Training",
                            "📂 Dataset Hub & Visual Inspector",
                            "🧪 Inference & Testing Playground",
                            "📦 Model Export Studio",
                            "🎛️ Hyperparameter Tuning",
                            "📊 Experiment History & Analytics"]
                    delegate: TabButton {
                        required property var modelData
                        required property int index
                        text: modelData
                        implicitHeight: 42
                        font.pixelSize: Theme.fsBody
                        font.bold: tabs.currentIndex === index
                        background: Rectangle {
                            color: tabs.currentIndex === index ? Theme.card : "transparent"
                            radius: 8
                            Rectangle {
                                anchors.bottom: parent.bottom
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: parent.width * 0.75
                                height: 2
                                radius: 1
                                color: tabs.currentIndex === index ? Theme.accent : "transparent"
                            }
                        }
                        contentItem: Text {
                            text: parent.text
                            font: parent.font
                            color: tabs.currentIndex === index ? Theme.text : Theme.textMuted
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                        }
                    }
                }
            }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: tabs.currentIndex

                TrainPage { }
                KagglePage { }
                DatasetPage { }
                InferencePage { }
                ExportPage { }
                TunePage { }
                HistoryPage { }
            }
        }
    }

    // ---- toast -----------------------------------------------------------
    Rectangle {
        id: toast
        property string kind: "info"
        function show(msg, k) {
            toastText.text = msg
            toast.kind = k || "info"
            toast.opacity = 1
            hideTimer.restart()
        }
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 20
        width: Math.min(560, toastText.implicitWidth + 34)
        height: toastText.implicitHeight + 24
        radius: 10
        opacity: 0
        visible: opacity > 0
        z: 999
        color: "#111c30"
        border.width: 1
        border.color: kind === "success" ? Theme.success
                    : kind === "warn" ? Theme.warn
                    : kind === "error" ? Theme.danger : Theme.accent
        Behavior on opacity { NumberAnimation { duration: 220 } }
        Text {
            id: toastText
            anchors.fill: parent
            anchors.margins: 12
            color: Theme.text
            font.pixelSize: Theme.fsBody
            wrapMode: Text.WordWrap
        }
        Timer { id: hideTimer; interval: 5200; onTriggered: toast.opacity = 0 }
        MouseArea { anchors.fill: parent; onClicked: toast.opacity = 0 }
    }
}
