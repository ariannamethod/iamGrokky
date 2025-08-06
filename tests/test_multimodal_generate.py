class DummyDW:
    def generate_response(self, prompt, api_key=None):
        return prompt


def test_generate_image_audio_different(monkeypatch):
    import importlib
    import jax

    monkeypatch.setattr(jax.config, "update", lambda *a, **k: None)
    model = importlib.import_module("SLNCX.model")

    monkeypatch.setattr(model, "analyze_image", lambda path: "img feature")
    monkeypatch.setattr(model, "analyze_audio", lambda path: "audio feature")
    monkeypatch.setattr(model, "DynamicWeights", lambda: DummyDW())

    text = "hello"
    out_img = model.generate(text, image="pic.png")
    out_audio = model.generate(text, audio="sound.wav")

    assert out_img != out_audio
    assert "img feature" in out_img
    assert "audio feature" in out_audio
