import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Studio

// Tab 3 — dataset statistics and the ground-truth annotation inspector.
Item {
    id: page

    function reloadSplit() {
        if (!App.hasDataset) return
        var split = splitCombo.currentText
        var paths = App.datasetInfo.splitPaths || ({})
        if (split && paths[split])
            Datasets.loadSplit(paths[split], App.datasetInfo.classNames || ({}))
    }

    Connections {
        target: App
        function onActiveDatasetChanged() {
            splitCombo.currentIndex = 0
            page.reloadSplit()
        }
    }
    Component.onCompleted: reloadSplit()

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
                title: "📂 Dataset Hub & Visual Ground-Truth Inspector"
                subtitle: "Inspect class balances, browse images, and verify bounding box annotations across splits."
            }

            InfoNote {
                text_: App.hasDataset ? "" :
                    "No active dataset selected. Choose or upload one in the sidebar."
                kind: "warn"
            }

            // A folder-per-class dataset trains under the classify task, and
            // Ultralytics needs the train/val split to exist on disk.
            InfoNote {
                text_: App.datasetInfo.kind === "classify"
                       ? "“" + App.datasetInfo.name + "” is a classification dataset ("
                         + (App.datasetInfo.classes || []).length
                         + " classes, folder per class). Train it with Task = classify — "
                         + "Ultralytics reads the folder directly, so there is no data.yaml "
                         + "to select."
                       : ""
                kind: "info"
            }

            Card {
                visible: App.datasetInfo.needsSplit === true
                SectionTitle {
                    title: "✂️ Create a train/val split"
                    subtitle: "This dataset has no split yet, which the classify task requires. "
                              + "Images are moved into train/ and val/ in place."
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Field {
                        label: "Validation share — " + Math.round(valSlider.value) + "%"
                        Layout.preferredWidth: 320
                        StudioSlider {
                            id: valSlider
                            from: 5; to: 40; value: 20; stepSize: 5
                        }
                    }
                    StudioButton {
                        text: App.busy !== "" ? "⏳ " + App.busy : "✂️ Create train/val split"
                        kind: "primary"
                        enabled: App.busy === ""
                        Layout.preferredWidth: 260
                        onClicked: App.splitActiveDataset(Math.round(valSlider.value))
                    }
                    Item { Layout.fillWidth: true }
                }
            }

            RowLayout {
                visible: App.hasDataset
                Layout.fillWidth: true
                spacing: 12

                // ---- statistics ----
                ColumnLayout {
                    Layout.preferredWidth: page.width * 0.32
                    Layout.fillWidth: false
                    Layout.alignment: Qt.AlignTop
                    spacing: 12

                    Card {
                        SectionTitle { title: "📊 Dataset Statistics" }
                        DataTable {
                            columns: [{ title: "Split", key: "split" },
                                      { title: "Image Count", key: "count", width: 120 }]
                            rows: App.datasetInfo.counts || []
                            maxHeight: 180
                        }
                    }
                    Card {
                        SectionTitle { title: "🏷️ Defined Classes" }
                        DataTable {
                            columns: [{ title: "Class ID", key: "id", width: 80 },
                                      { title: "Class Name", key: "name" }]
                            rows: App.datasetInfo.classes || []
                            maxHeight: 260
                        }
                    }
                    Card {
                        SectionTitle { title: "📄 Config File" }
                        Text {
                            text: App.datasetInfo.yaml || ""
                            color: Theme.success
                            font.family: Theme.mono
                            font.pixelSize: Theme.fsSmall
                            wrapMode: Text.WrapAnywhere
                            Layout.fillWidth: true
                        }
                        StudioButton {
                            text: "📂 Open dataset folder"
                            kind: "ghost"
                            Layout.fillWidth: true
                            onClicked: App.revealPath(App.datasetInfo.yaml)
                        }
                    }
                }

                // ---- inspector ----
                Card {
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignTop
                    SectionTitle { title: "🖼️ Visual Annotation Inspector" }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Field {
                            label: "Split to Inspect"
                            Layout.preferredWidth: 220
                            StudioCombo {
                                id: splitCombo
                                model: App.datasetInfo.splits || []
                                onActivated: page.reloadSplit()
                            }
                        }
                        Field {
                            label: "Browse Image Index — " + (idxSlider.value + 1)
                                   + " / " + Datasets.imageCount
                            StudioSlider {
                                id: idxSlider
                                from: 0
                                to: Math.max(0, Datasets.imageCount - 1)
                                stepSize: 1
                                snapMode: Slider.SnapAlways
                                onMoved: Datasets.showIndex(value, App.datasetInfo.classNames || ({}))
                            }
                        }
                        StudioButton {
                            text: "◀"
                            Layout.preferredWidth: 42
                            onClicked: { idxSlider.value = Math.max(0, idxSlider.value - 1)
                                         Datasets.showIndex(idxSlider.value, App.datasetInfo.classNames || ({})) }
                        }
                        StudioButton {
                            text: "▶"
                            Layout.preferredWidth: 42
                            onClicked: { idxSlider.value = Math.min(idxSlider.to, idxSlider.value + 1)
                                         Datasets.showIndex(idxSlider.value, App.datasetInfo.classNames || ({})) }
                        }
                    }

                    InfoNote {
                        text_: Datasets.imageCount === 0 && App.hasDataset
                               ? "No images found in this split on disk." : ""
                        kind: "warn"
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 520
                        radius: 10
                        color: "#05080f"
                        border.color: Theme.border
                        border.width: 1
                        clip: true
                        Image {
                            anchors.fill: parent
                            anchors.margins: 6
                            fillMode: Image.PreserveAspectFit
                            cache: false
                            asynchronous: true
                            source: Datasets.previewPath ? "file://" + Datasets.previewPath : ""
                        }
                    }
                    Text {
                        text: Datasets.currentName
                              ? "Image [" + (idxSlider.value + 1) + "/" + Datasets.imageCount + "]: "
                                + Datasets.currentName + " • Ground Truth Boxes: " + Datasets.boxCount
                              : ""
                        color: Theme.textMuted
                        font.pixelSize: Theme.fsSmall
                        Layout.fillWidth: true
                        elide: Text.ElideMiddle
                    }
                }
            }
            Item { Layout.preferredHeight: 10 }
        }
    }
}
