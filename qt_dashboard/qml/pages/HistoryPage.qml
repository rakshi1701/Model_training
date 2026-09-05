import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Studio

// Tab 7 — run history table, multi-run overlays and per-run figures.
Item {
    id: page
    property var selected: []

    function toggle(name) {
        var s = selected.slice()
        var i = s.indexOf(name)
        if (i >= 0) s.splice(i, 1); else s.push(name)
        selected = s
        History.buildOverlay(selected, metricCombo.currentText)
    }

    Component.onCompleted: {
        History.refresh()
        var names = History.runNames
        selected = names.slice(0, Math.min(3, names.length))
        History.buildOverlay(selected, "metrics/mAP50(B)")
    }

    FileDialog {
        id: saveDialog
        property string source: ""
        title: "Save weights as…"
        fileMode: FileDialog.SaveFile
        nameFilters: ["PyTorch weights (*.pt)"]
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

            RowLayout {
                Layout.fillWidth: true
                SectionTitle {
                    title: "📊 Training Run History & Cross-Experiment Comparison"
                    subtitle: "Every experiment under yolo_workspace/runs/, with overlay curves and artifacts."
                }
                StudioButton {
                    text: "🔄 Refresh"
                    onClicked: { History.refresh(); App.refreshModels() }
                }
            }

            InfoNote {
                text_: History.runs.length === 0
                       ? "No training runs found in yolo_workspace/runs/." : ""
                kind: "info"
            }

            Card {
                visible: History.runs.length > 0
                SectionTitle { title: "Experiments" }
                DataTable {
                    columns: [{ title: "Run Name", key: "name" },
                              { title: "Epochs", key: "epochs", width: 90 },
                              { title: "Best mAP@50", key: "map50", width: 130 },
                              { title: "Best mAP@50-95", key: "map5095", width: 150 },
                              { title: "Kaggle kernel", key: "owner", width: 220 }]
                    rows: History.runs
                    maxHeight: 300
                }
            }

            Card {
                visible: History.runs.length > 0
                SectionTitle { title: "🔍 Multi-Run Curve Overlay" }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Field {
                        label: "Metric to Compare"
                        Layout.preferredWidth: 320
                        StudioCombo {
                            id: metricCombo
                            model: History.compareMetrics
                            onActivated: History.buildOverlay(page.selected, currentText)
                        }
                    }
                    Item { Layout.fillWidth: true }
                }

                Flow {
                    Layout.fillWidth: true
                    spacing: 8
                    Repeater {
                        model: History.runs
                        delegate: Rectangle {
                            required property var modelData
                            property bool on: page.selected.indexOf(modelData.name) >= 0
                            height: 28
                            width: chipText.implicitWidth + 26
                            radius: 14
                            color: on ? Qt.rgba(0.22, 0.74, 0.97, 0.16) : Theme.card
                            border.width: 1
                            border.color: on ? Theme.borderStrong : Theme.border
                            Text {
                                id: chipText
                                anchors.centerIn: parent
                                text: (on ? "✓ " : "") + modelData.name
                                color: on ? Theme.accent : Theme.textMuted
                                font.pixelSize: Theme.fsSmall
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: page.toggle(modelData.name)
                            }
                        }
                    }
                }

                MetricChart {
                    seriesData: History.overlay
                    yTitle: metricCombo.currentText
                    implicitHeight: 360
                    emptyText: "Pick one or more runs above to overlay their curves."
                }
            }

            // ---- per-run artifacts ----
            Card {
                visible: History.runs.length > 0
                SectionTitle { title: "🖼️ Run Artifacts" }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Field {
                        label: "Run"
                        Layout.preferredWidth: 320
                        StudioCombo {
                            id: runCombo
                            model: History.runNames
                            onActivated: figures.model = History.runFigures(currentText)
                        }
                    }
                    StudioButton {
                        text: "📂 Open run folder"
                        kind: "ghost"
                        onClicked: {
                            for (var i = 0; i < History.runs.length; ++i)
                                if (History.runs[i].name === runCombo.currentText)
                                    App.revealPath(History.runs[i].dir)
                        }
                    }
                    StudioButton {
                        text: "📥 Save best.pt…"
                        onClicked: {
                            for (var i = 0; i < History.runs.length; ++i) {
                                if (History.runs[i].name === runCombo.currentText
                                        && History.runs[i].weights !== "") {
                                    saveDialog.source = History.runs[i].weights
                                    saveDialog.open()
                                    return
                                }
                            }
                            Notifier.notify("That run has no best.pt on disk.", "warn")
                        }
                    }
                    StudioButton {
                        text: "🖼️ Load figures"
                        onClicked: figures.model = History.runFigures(runCombo.currentText)
                    }
                }

                Flow {
                    Layout.fillWidth: true
                    spacing: 10
                    Repeater {
                        id: figures
                        model: []
                        delegate: ColumnLayout {
                            required property var modelData
                            width: 300
                            spacing: 3
                            Rectangle {
                                Layout.preferredWidth: 300
                                Layout.preferredHeight: 220
                                radius: 8
                                color: "#05080f"
                                border.color: Theme.border
                                border.width: 1
                                clip: true
                                Image {
                                    anchors.fill: parent
                                    anchors.margins: 4
                                    fillMode: Image.PreserveAspectFit
                                    asynchronous: true
                                    cache: false
                                    source: "file://" + modelData.path
                                }
                            }
                            Text {
                                text: modelData.name
                                color: Theme.textMuted
                                font.pixelSize: Theme.fsTiny
                                elide: Text.ElideMiddle
                                Layout.preferredWidth: 300
                            }
                        }
                    }
                }
            }
            Item { Layout.preferredHeight: 10 }
        }
    }
}
