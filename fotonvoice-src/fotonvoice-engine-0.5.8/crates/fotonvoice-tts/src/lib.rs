//! FotonVoice Engine text-to-speech engine.

mod audiocpp;
pub mod breeze;
mod engine;
pub mod hf;
mod fifo;
pub mod inflect;
pub mod luxtts;
mod piper;
mod pocket;
pub mod voxcpm;

pub use breeze::{
    breeze_tts_2_model_dir, download_breeze_tts_2_assets, is_breeze_tts_2_ready,
};
pub use voxcpm::{
    download_vox_cpm_2_assets, is_vox_cpm_2_ready, vox_cpm_2_model_dir,
};
pub use voxcpm::is_vox_cpm_2_ready as is_vox_cpm_2_downloaded;
pub use engine::{
    stop_current_playback, ErrorCallback, PlaybackCallback, TtsCommand, TtsEngineHandle,
    TtsEngineWorker, Utterance,
};
pub use fifo::run_fifo_responder;
pub use hf::{
    apply_hf_token, classify_download_error, effective_hf_token, hf_token_from_env,
    looks_like_auth_failure, token_rejected, HF_TOKEN_ENV, HF_TOKEN_REJECTED_TAG,
};
pub use inflect::{
    download_inflect_micro_assets, inflect_micro_model_dir, is_inflect_micro_downloaded,
    INFLECT_MICRO_COMPILED,
};
pub use luxtts::{is_lux_tts_downloaded, lux_tts_model_dir, LUX_TTS_COMPILED};
pub use piper::{
    download_piper_binary, download_voice, get_voice_path, is_voice_downloaded, list_local_voices,
    piper_binary, piper_voices_dir, VoiceInfo, PIPER_VOICES,
};
pub use pocket::{
    cloned_tts_voices_dir, download_pocket_tts_assets, is_pocket_tts_ready, pocket_tts_voice,
    pocket_tts_voice_catalogue, PocketTtsVoiceInfo, PocketTtsVoiceOption, POCKET_TTS_VOICES,
};

pub use fotonvoice_text::{correct_custom_vocabulary, expand_snippets};
