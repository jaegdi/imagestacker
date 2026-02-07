use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMode {
    StartFresh,  // Delete existing aligned images and start from scratch
    Resume,      // Continue from where we left off (skip already aligned images)
}

#[derive(Debug, Clone)]
pub enum Message {
    AddFolder,
    AddImages,
    AlignImages,
    AlignImagesConfirmed(Option<AlignmentMode>),
    AlignmentDone(Result<opencv::core::Rect, String>),
    AutoRefreshTick,
    CancelAlignedSelection,
    CancelBunchSelection,
    CancelImportedSelection,
    CancelSharpnessSelection,
    CloseImagePreview,
    DeletePreviewImage,
    DeselectAllAligned,
    DeselectAllBunches,
    DeselectAllImported,
    DeselectAllSharpness,
    DetectSharpness,
    ImagePreviewLoaded(PathBuf, iced::widget::image::Handle, bool),
    ImagesSelected(Vec<PathBuf>),
    InternalPathsScanned(Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>),  // imported, sharpness, aligned, bunches, final
    LoadFolder(PathBuf),  // New: Load folder directly without dialog
    LoadFullImage(PathBuf),
    NavigationThrottleReset,
    NextImageInPreview,
    OpenImage(PathBuf),
    OpenImageWithExternalEditor(PathBuf),
    PreviousImageInPreview,
    RefreshAlignedPane,
    RefreshBunchesPane,
    RefreshFinalPane,
    RefreshImportedPane,
    RefreshPanes,
    RefreshSharpnessPane,
    SelectAllAligned,
    SelectAllBunches,
    SelectAllImported,
    SelectAllSharpness,
    SharpnessDetectionDone(Result<Vec<PathBuf>, String>),  // YAML file paths
    ShowImagePreview(PathBuf, Vec<PathBuf>), // path and list of all images in pane
    StackBunches,
    StackImages,
    StackImported,
    StackSharpness,
    StackingDone(Result<(Vec<u8>, opencv::core::Mat), String>),
    StackSelectedAligned,
    StackSelectedBunches,
    StackSelectedImported,
    StackSelectedSharpness,
    ThumbnailUpdated(PathBuf, iced::widget::image::Handle),
    ToggleAlignedImage(PathBuf),
    ToggleBunchImage(PathBuf),
    ToggleImportedImage(PathBuf),
    ToggleSharpnessImage(PathBuf),
    // New: Configuration messages
    CloseHelp,
    CloseLog,
    FeatureDetectorChanged(crate::config::FeatureDetector),
    ResetToDefaults,
    SharpnessGridSizeChanged(f32),
    SharpnessIqrMultiplierChanged(f32),
    SharpnessThresholdChanged(f32),
    ToggleHelp,
    ToggleLog,
    ToggleSettings,
    UseAdaptiveBatchSizes(bool),
    UseCLAHE(bool),
    // ECC-specific parameters
    EccBatchSizeChanged(f32),
    EccChunkSizeChanged(f32),
    EccEpsilonChanged(f32),
    EccGaussFilterSizeChanged(f32),
    EccMaxIterationsChanged(f32),
    EccMotionTypeChanged(crate::config::EccMotionType),
    EccUseHybridChanged(bool),
    EccTimeoutChanged(f32),
    // Transform validation parameters
    MaxTransformScaleChanged(f32),
    MaxTransformTranslationChanged(f32),
    MaxTransformDeterminantChanged(f32),
    ProgressUpdate(String, f32),
    // Advanced processing options
    BrightnessBoostChanged(f32),
    ContrastBoostChanged(f32),
    EnableColorCorrection(bool),
    EnableNoiseReduction(bool),
    EnableSharpening(bool),
    NoiseReductionStrengthChanged(f32),
    SaturationBoostChanged(f32),
    SharpeningStrengthChanged(f32),
    // Preview settings
    ExternalEditorPathChanged(String),
    ExternalViewerPathChanged(String),
    PreviewMaxHeightChanged(f32),
    PreviewMaxWidthChanged(f32),
    UseInternalPreview(bool),
    // GPU / Performance settings
    GpuConcurrencyChanged(f32),
    // Stacking bunch size settings
    AutoBunchSizeChanged(bool),
    BunchSizeChanged(f32),
    // Font settings
    DefaultFontChanged(String),
    // Scroll position tracking
    AlignedScrollChanged(f32),
    BunchesScrollChanged(f32),
    FinalScrollChanged(f32),
    ImportedScrollChanged(f32),
    SharpnessScrollChanged(f32),
    // Window resize for responsive layout
    WindowResized(f32, f32),
    // PaneGrid events
    PaneResized(iced::widget::pane_grid::ResizeEvent),
    PaneDragged(iced::widget::pane_grid::DragEvent),
    Exit,
    None,
}