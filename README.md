# 🎙️ Voice to Order

![Status](https://img.shields.io/badge/status-in%20progress-orange?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Deepgram](https://img.shields.io/badge/Deepgram-13EF93?style=for-the-badge&logo=deepgram&logoColor=white) ![Whisper](https://img.shields.io/badge/Whisper-412991?style=for-the-badge&logo=openai&logoColor=white) ![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)

> Turn customer voicemails into structured orders using several speech-to-text providers and an LLM consensus step.

## 🎯 Why this project

Voicemail orders are noisy: accents, product codes and background sound all cause transcription errors. This pipeline transcribes each message with multiple providers, reconciles the transcripts with an LLM, and then extracts the customer, order lines and delivery date.

## 🧱 Planned stack

- Python 3.12
- FFmpeg for audio conversion
- Deepgram and Whisper for speech-to-text
- LLM consensus to reconcile transcripts
- FastAPI endpoint and background worker

## 🗺️ Roadmap

- [ ] Audio ingestion and conversion to MP3/WAV
- [ ] Parallel transcription with multiple providers
- [ ] Transcript consensus prompt
- [ ] Order extraction from the final transcript
- [ ] Majority vote for delivery dates
- [ ] Accuracy comparison per provider

## 📌 Status

🚧 This project is in early development. Code is coming soon. It uses synthetic sample data only.

---

Built by [Jahanzaib Ahmad](https://github.com/jehanxaibahmed) · Full Stack Engineer · AI & LLM Systems
