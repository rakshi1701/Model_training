import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Studio

// Tab 2 — Kaggle cloud GPU hub: credentials, quota, dispatch and the job board.
Item {
    id: page

    property string filterCat: "All"
    property string search: ""
    property var expanded: ({})
    property var autoExport: []

    function shown() {
        var out = []
        var jobs = Kaggle ? Kaggle.jobs : []
        for (var i = 0; i < jobs.length; ++i) {
            var j = jobs[i]
            if (filterCat !== "All" && j.catName !== filterCat) continue
            if (search !== "") {
                var q = search.toLowerCase()
                if (j.ref.toLowerCase().indexOf(q) < 0
                        && String(j.model_name).toLowerCase().indexOf(q) < 0) continue
            }
            out.push(j)
        }
        return out
    }

    function toggleExport(fmt) {
        var a = autoExport.slice()
        var i = a.indexOf(fmt)
        if (i >= 0) a.splice(i, 1); else a.push(fmt)
        autoExport = a
        if (Kaggle) Kaggle.setAutoExport(a)
    }

    FileDialog {
        id: jsonDialog
        title: "Select kaggle.json"
        nameFilters: ["Kaggle token (*.json)"]
        onAccepted: Kaggle.importKaggleJson(selectedFile)
    }
    FileDialog {
        id: saveDialog
        property string source: ""
        title: "Save file as…"
        fileMode: FileDialog.SaveFile
        onAccepted: App.saveCopy(source, selectedFile)
    }

    Component.onCompleted: {
        if (Kaggle) {
            targetExpField.text = Kaggle.nextRunName()
            resumeCombo.model = ["(start a fresh run)"].concat(Kaggle.resumableJobs())
        }
    }

    ScrollView {
        objectName: "kaggleScroll"
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: page.width - 28
            x: 14
            y: 12
            spacing: 12

            SectionTitle {
                title: "☁️ Kaggle Cloud GPU Training Hub"
                subtitle: "Offload training to Kaggle's free cluster (dual Tesla T4 32GB / P100) and sync models back automatically."
            }

            InfoNote {
                text_: Kaggle ? "" : ("Kaggle integration unavailable — " + KaggleError)
                kind: "error"
            }

            ColumnLayout {
                visible: Kaggle !== null
                Layout.fillWidth: true
                spacing: 12

                // ---------------- 1. auth ----------------
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12

                    Card {
                        objectName: "authCard"
                        Layout.preferredWidth: page.width * 0.42
                        Layout.fillWidth: false
                        Layout.alignment: Qt.AlignTop
                        accentColor: Kaggle && Kaggle.auth.connected
                                     ? Qt.rgba(0.06, 0.72, 0.51, 0.35)
                                     : Qt.rgba(0.96, 0.62, 0.04, 0.35)
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: Kaggle && Kaggle.auth.connected
                                      ? "🟢 Kaggle API Connected: @" + Kaggle.auth.user
                                      : "⚠️ Kaggle API Not Connected"
                                color: Kaggle && Kaggle.auth.connected ? Theme.green : Theme.warn
                                font.pixelSize: Theme.fsTitle
                                font.bold: true
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                            StudioButton {
                                text: "🔄"
                                Layout.preferredWidth: 40
                                onClicked: Kaggle.refreshAuth()
                            }
                        }
                        Text {
                            visible: !(Kaggle && Kaggle.auth.connected)
                            text: (Kaggle && Kaggle.auth.error) ? Kaggle.auth.error
                                  : "Configure your API token to enable 1-click cloud GPU training."
                            color: Theme.textDim
                            font.pixelSize: Theme.fsSmall
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }

                        // Signing out deletes the token files on this machine.
                        ColumnLayout {
                            id: signOut
                            Layout.fillWidth: true
                            visible: Kaggle && Kaggle.auth.connected
                            spacing: 6
                            property bool confirming: false

                            StudioButton {
                                text: "🚪 Disconnect account"
                                visible: !signOut.confirming
                                Layout.fillWidth: true
                                onClicked: signOut.confirming = true
                            }
                            InfoNote {
                                visible: signOut.confirming
                                kind: "warn"
                                text_: "Remove the stored credentials for @"
                                       + (Kaggle ? Kaggle.auth.user : "") + "?\n"
                                       + (Kaggle && Kaggle.credentialPaths.length
                                          ? "Deletes: " + Kaggle.credentialPaths.join("\n         ")
                                          : "No credential files found — this clears the session only.")
                                       + "\n\nThe token keeps working elsewhere until you revoke it "
                                       + "at kaggle.com/settings."
                            }
                            RowLayout {
                                visible: signOut.confirming
                                Layout.fillWidth: true
                                spacing: 6
                                StudioButton {
                                    text: "🚪 Disconnect"
                                    kind: "danger"
                                    Layout.fillWidth: true
                                    onClicked: {
                                        signOut.confirming = false
                                        Kaggle.disconnectAccount()
                                    }
                                }
                                StudioButton {
                                    text: "🔑 Revoke on kaggle.com"
                                    Layout.fillWidth: true
                                    onClicked: App.openUrl("https://www.kaggle.com/settings")
                                }
                                StudioButton {
                                    text: "Cancel"
                                    Layout.fillWidth: true
                                    onClicked: signOut.confirming = false
                                }
                            }
                        }
                    }

                    Card {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        SectionTitle {
                            title: "🔑 Kaggle API Credentials"
                            subtitle: "kaggle.com → Account Settings → API → Create New Token"
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            StudioButton {
                                text: "📤 Import kaggle.json"
                                Layout.fillWidth: true
                                onClicked: jsonDialog.open()
                            }
                            StudioButton {
                                text: "🌐 Open Kaggle settings"
                                kind: "ghost"
                                Layout.fillWidth: true
                                onClicked: App.openUrl("https://www.kaggle.com/settings")
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Field {
                                label: "Username (handle, not email)"
                                StudioText {
                                    id: userField
                                    text: Kaggle ? Kaggle.auth.user : ""
                                    placeholderText: "e.g. rakshithr1701"
                                }
                            }
                            Field {
                                label: "API Key"
                                StudioText {
                                    id: keyField
                                    echoMode: TextInput.Password
                                    placeholderText: "e.g. 38d9c…"
                                }
                            }
                        }
                        StudioButton {
                            text: "💾 Save Credentials & Connect"
                            kind: "primary"
                            Layout.fillWidth: true
                            onClicked: Kaggle.saveCredentials(userField.text, keyField.text)
                        }
                    }
                }

                // ---------------- 2. quota ----------------
                Card {
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        MetricCard {
                            label: "⚡ Remote GPU Cluster"
                            value: "2x T4 / P100"
                            delta: "32GB VRAM (DDP)"
                        }
                        MetricCard {
                            label: "⏱️ Est. GPU Time Used"
                            inverse: true
                            value: Kaggle ? (Kaggle.quota.used_hours || 0).toFixed(1) + " h" : "—"
                            delta: Kaggle ? Math.round(Kaggle.quota.pct_used || 0) + "% of "
                                            + Math.round(Kaggle.quota.quota_hours || 30) + "h weekly" : ""
                        }
                        MetricCard {
                            label: "🔋 Est. Remaining"
                            valueColor: Theme.success
                            value: Kaggle ? (Kaggle.quota.remaining_hours || 0).toFixed(1) + " h" : "—"
                            delta: "rolling 7 days"
                        }
                        MetricCard {
                            label: "📦 Local Ingestion"
                            value: "Auto-Sync"
                            delta: "best.pt & results.csv"
                        }
                    }
                    ProgressBarLine {
                        value: Kaggle ? Math.min(1, (Kaggle.quota.pct_used || 0) / 100) : 0
                        caption: Kaggle ? (Kaggle.quota.note || "") : ""
                        barColor: (Kaggle && (Kaggle.quota.pct_used || 0) > 80) ? Theme.danger : Theme.success
                    }
                }

                // ---------------- 3. dispatcher ----------------
                Card {
                    SectionTitle { title: "🚀 Configure Remote Training Job" }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                        spacing: 18

                        // dataset & identification
                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignTop
                            spacing: 6
                            Text {
                                text: "1. Dataset & Identification"
                                color: Theme.accent
                                font.pixelSize: Theme.fsSmall
                                font.bold: true
                            }
                            Field {
                                label: "Dataset to Train On"
                                StudioCombo {
                                    id: kDsCombo
                                    model: App.datasetNames
                                    onActivated: App.selectDataset(App.datasetNames[currentIndex])
                                    function sync() {
                                        var i = App.datasetNames.indexOf(App.activeDatasetKey)
                                        if (i >= 0 && i !== currentIndex)
                                            currentIndex = i
                                    }
                                    Component.onCompleted: sync()
                                    Connections {
                                        target: App
                                        function onActiveDatasetChanged() { kDsCombo.sync() }
                                        function onDatasetsChanged() { kDsCombo.sync() }
                                    }
                                }
                            }
                            Field {
                                label: "Kaggle Dataset Title"
                                hint: "Remote dataset name on Kaggle"
                                StudioText {
                                    id: kDsTitle
                                    text: App.datasetInfo.name || "yolo-dataset"
                                    Connections {
                                        target: App
                                        function onActiveDatasetChanged() {
                                            kDsTitle.text = App.datasetInfo.name || "yolo-dataset"
                                        }
                                    }
                                }
                            }
                            Field {
                                label: "Or use existing Kaggle dataset ref(s)"
                                hint: "Comma-separated owner/slug refs — skips uploading and attaches these directly."
                                StudioText {
                                    id: kExisting
                                    placeholderText: "user/my-dataset  or  user/ds-p0, user/ds-p1"
                                }
                            }
                            Field {
                                label: "Kaggle Kernel Job Title"
                                StudioText { id: kJobTitle; text: "yolo11-cloud-training" }
                            }
                            Field {
                                label: "Local Target Run Name"
                                hint: "Folder under yolo_workspace/runs/ — auto-incremented so runs never overwrite each other."
                                StudioText { id: targetExpField }
                            }
                        }

                        // hyperparameters
                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignTop
                            spacing: 6
                            Text {
                                text: "2. Remote Hyperparameters"
                                color: Theme.accent
                                font.pixelSize: Theme.fsSmall
                                font.bold: true
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Field {
                                    label: "Model Architecture"
                                    StudioCombo {
                                        id: kFamily
                                        model: App.modelFamilies
                                        onActivated: kWeights.model = App.kaggleWeights(currentText)
                                    }
                                }
                                Field {
                                    label: "Weights / Scale"
                                    StudioCombo {
                                        id: kWeights
                                        model: App.kaggleWeights("YOLO11")
                                    }
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Field {
                                    label: "Epochs — " + Math.round(kEpochs.value)
                                    StudioSlider { id: kEpochs; from: 1; to: 300; value: 100; stepSize: 5 }
                                }
                                Field {
                                    label: "Optimizer"
                                    StudioCombo { id: kOpt; model: ["AdamW", "SGD", "Adam", "auto"] }
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Field {
                                    label: "Batch Size (GPU)"
                                    StudioCombo {
                                        id: kBatch
                                        model: ["8", "16", "32", "64", "128"]
                                        currentIndex: 2
                                    }
                                }
                                Field {
                                    label: "Image Size (imgsz)"
                                    StudioCombo {
                                        id: kImgsz
                                        model: ["320", "416", "512", "640", "768", "1024", "1280"]
                                        currentIndex: 3
                                    }
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                Field {
                                    label: "Initial LR (lr0)"
                                    StudioSpin {
                                        id: kLr
                                        decimals: 4; realFrom: 0.0001; realTo: 1.0; realStep: 0.001
                                        value: 50
                                    }
                                }
                                Field {
                                    label: "Early Stopping Patience"
                                    StudioSpin { id: kPatience; realFrom: 0; realTo: 100; value: 20 }
                                }
                            }
                            StudioCheck {
                                id: kDualGpu
                                text: "⚡ Leverage Dual-GPU Distributed Data Parallel (2x T4)"
                                checked: true
                            }
                            Field {
                                label: "⏳ Max runtime — " + kMaxHours.value.toFixed(1) + " h"
                                hint: "Kaggle kills a session at 12h; training stops at this cap and packages last.pt."
                                StudioSlider {
                                    id: kMaxHours
                                    from: 1.0; to: 11.5; value: 11.0; stepSize: 0.5
                                }
                            }
                            Field {
                                label: "♻️ Resume from previous job"
                                hint: "Mounts that kernel's output and continues from its last.pt."
                                StudioCombo {
                                    id: resumeCombo
                                    model: ["(start a fresh run)"]
                                }
                            }
                        }
                    }

                    Text {
                        text: "3. After the job finishes"
                        color: Theme.accent
                        font.pixelSize: Theme.fsSmall
                        font.bold: true
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 18
                        StudioCheck {
                            id: kAutoIngest
                            text: "📥 Auto-ingest results when the job completes"
                            checked: true
                            onCheckedChanged: if (Kaggle) Kaggle.setAutoIngest(checked)
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Text {
                                text: "📦 Auto-export after ingest:"
                                color: Theme.textMuted
                                font.pixelSize: Theme.fsSmall
                            }
                            Repeater {
                                model: ["onnx", "torchscript", "openvino"]
                                delegate: Rectangle {
                                    required property var modelData
                                    property bool on: page.autoExport.indexOf(modelData) >= 0
                                    height: 26
                                    width: fmtLabel.implicitWidth + 24
                                    radius: 13
                                    color: on ? Qt.rgba(0.22, 0.74, 0.97, 0.16) : Theme.card
                                    border.width: 1
                                    border.color: on ? Theme.borderStrong : Theme.border
                                    Text {
                                        id: fmtLabel
                                        anchors.centerIn: parent
                                        text: (on ? "✓ " : "") + modelData
                                        color: on ? Theme.accent : Theme.textMuted
                                        font.pixelSize: Theme.fsSmall
                                    }
                                    MouseArea {
                                        anchors.fill: parent
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: page.toggleExport(modelData)
                                    }
                                }
                            }
                            Item { Layout.fillWidth: true }
                        }
                    }

                    StudioButton {
                        text: (Kaggle && Kaggle.busy !== "") ? "⏳ " + Kaggle.busy
                              : "🚀 Stage Dataset & Dispatch Training to Kaggle GPU"
                        kind: "primary"
                        Layout.fillWidth: true
                        enabled: Kaggle && Kaggle.auth.connected && Kaggle.busy === ""
                                 && (App.hasDataset || kExisting.text !== "")
                        onClicked: Kaggle.dispatch({
                            "datasetYaml": App.dataYamlPath,
                            "datasetTitle": kDsTitle.text,
                            "existingRefs": kExisting.text,
                            "jobTitle": kJobTitle.text,
                            "targetExp": targetExpField.text,
                            "model": kWeights.currentText,
                            "epochs": Math.round(kEpochs.value),
                            "batch": parseInt(kBatch.currentText),
                            "imgsz": parseInt(kImgsz.currentText),
                            "optimizer": kOpt.currentText,
                            "lr0": kLr.realValue,
                            "patience": kPatience.value,
                            "dualGpu": kDualGpu.checked,
                            "maxHours": kMaxHours.value,
                            "resumeFrom": resumeCombo.currentIndex === 0 ? "" : resumeCombo.currentText
                        })
                    }
                    LogView {
                        visible: Kaggle && Kaggle.dispatchLog.length > 0
                        text_: Kaggle ? Kaggle.dispatchLog.join("\n") : ""
                        implicitHeight: 150
                    }
                }

                // ---------------- 4. job dashboard ----------------
                Card {
                    RowLayout {
                        Layout.fillWidth: true
                        SectionTitle {
                            title: "📡 Kaggle Training Jobs"
                            subtitle: Kaggle && Kaggle.livePolling
                                      ? "🔄 Live — last polled " + Kaggle.lastPolled + " (every 30s)"
                                      : "Listed from local history and your Kaggle account, newest first."
                        }
                        StudioSwitch {
                            text: "🔄 Live polling"
                            checked: Kaggle ? Kaggle.livePolling : false
                            onToggled: Kaggle.setLivePolling(checked)
                        }
                        StudioButton {
                            text: "🔄 Refresh Jobs"
                            onClicked: { Kaggle.refreshJobs(); Kaggle.refreshQuota() }
                        }
                    }

                    LogView {
                        visible: Kaggle && Kaggle.ingestLog.length > 0
                        text_: Kaggle ? Kaggle.ingestLog.join("\n") : ""
                        implicitHeight: 120
                    }

                    // status counts
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Repeater {
                            model: Kaggle ? Kaggle.categoryLabels : []
                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: 58
                                radius: 10
                                color: Theme.card
                                border.width: 1
                                border.color: Theme.border
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 8
                                    spacing: 2
                                    Text {
                                        text: modelData.icon + " " + modelData.name
                                        color: Theme.textMuted
                                        font.pixelSize: Theme.fsTiny
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                    Text {
                                        text: Kaggle ? (Kaggle.counts[modelData.key] || 0) : 0
                                        color: modelData.colour
                                        font.pixelSize: 19
                                        font.bold: true
                                    }
                                }
                                ToolTip.visible: hover.hovered
                                ToolTip.text: modelData.help
                                HoverHandler { id: hover }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Field {
                            label: "Show"
                            Layout.preferredWidth: 260
                            StudioCombo {
                                id: filterCombo
                                model: {
                                    var out = ["All"]
                                    var cats = Kaggle ? Kaggle.categoryLabels : []
                                    for (var i = 0; i < cats.length; ++i) {
                                        var n = Kaggle.counts[cats[i].key] || 0
                                        if (n > 0) out.push(cats[i].name)
                                    }
                                    return out
                                }
                                onActivated: page.filterCat = currentText
                            }
                        }
                        Field {
                            label: "Search job name or model"
                            StudioText {
                                placeholderText: "e.g. yolov8, cctv, 20260904"
                                onTextChanged: page.search = text
                            }
                        }
                    }

                    Text {
                        text: Kaggle ? ("Showing " + page.shown().length + " of "
                                        + Kaggle.jobs.length + " job(s), newest first.") : ""
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsSmall
                    }
                    InfoNote {
                        text_: (Kaggle && Kaggle.jobs.length === 0)
                               ? "No training jobs found. Launch one above and it will appear here." : ""
                        kind: "info"
                    }

                    Repeater {
                        model: page.shown()
                        delegate: JobCard {
                            required property var modelData
                            job: modelData
                            Layout.fillWidth: true
                            onSaveWeight: function(path) {
                                saveDialog.source = path
                                saveDialog.open()
                            }
                        }
                    }
                }

                // ---------------- 5. notebook template ----------------
                Card {
                    SectionTitle {
                        title: "📓 Standalone Kaggle Notebook Template"
                        subtitle: "Prefer the Kaggle UI? Save this notebook, upload it, attach your dataset and hit Run All."
                    }
                    StudioButton {
                        text: "📥 Save kaggle_yolo_train_template.ipynb…"
                        Layout.fillWidth: true
                        enabled: Kaggle && Kaggle.notebookTemplate() !== ""
                        onClicked: {
                            saveDialog.source = Kaggle.notebookTemplate()
                            saveDialog.open()
                        }
                    }
                }
            }
            Item { Layout.preferredHeight: 10 }
        }
    }
}
