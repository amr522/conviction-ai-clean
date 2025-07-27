import pytest
import json
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError

from src.register_schema import register_schema, load_schema


@pytest.fixture
def fake_schema(tmp_path):
    """Create a temporary schema file for testing."""
    schema = {
        "type": "object", 
        "properties": {
            "foo": {"type": "string"},
            "bar": {"type": "number"}
        }
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema))
    return str(path)


def test_load_schema(fake_schema):
    """Test schema loading from file."""
    schema = load_schema(fake_schema)
    assert schema["type"] == "object"
    assert "foo" in schema["properties"]


@patch("boto3.client")
def test_create_new_schema(mock_boto, fake_schema):
    """Test creating a new schema in AWS Glue."""
    client = MagicMock()
    client.create_schema.return_value = {
        "SchemaArn": "arn:aws:glue:us-east-1:123456789012:schema/test-registry/test-schema"
    }
    mock_boto.return_value = client

    register_schema("test-registry", "test-schema", fake_schema, "BACKWARD")
    
    client.create_schema.assert_called_once()
    client.update_schema.assert_not_called()
    
    # Verify call arguments
    call_args = client.create_schema.call_args[1]
    assert call_args["RegistryId"]["RegistryName"] == "test-registry"
    assert call_args["SchemaName"] == "test-schema"
    assert call_args["Compatibility"] == "BACKWARD"


@patch("boto3.client")
def test_update_existing_schema(mock_boto, fake_schema):
    """Test updating an existing schema."""
    client = MagicMock()
    
    # Mock AlreadyExistsException on create
    err = ClientError(
        {"Error": {"Code": "AlreadyExistsException"}}, 
        "CreateSchema"
    )
    client.create_schema.side_effect = err
    client.update_schema.return_value = {"SchemaVersionNumber": "2"}
    mock_boto.return_value = client

    register_schema("test-registry", "test-schema", fake_schema, "BACKWARD")
    
    client.create_schema.assert_called_once()
    client.update_schema.assert_called_once()
    
    # Verify update call arguments
    call_args = client.update_schema.call_args[1]
    assert call_args["RegistryId"]["RegistryName"] == "test-registry"
    assert call_args["SchemaName"] == "test-schema"
    assert call_args["Compatibility"] == "BACKWARD"


@patch("boto3.client")
def test_fail_on_access_denied(mock_boto, fake_schema):
    """Test failure on access denied error."""
    client = MagicMock()
    err = ClientError(
        {"Error": {"Code": "AccessDeniedException"}}, 
        "CreateSchema"
    )
    client.create_schema.side_effect = err
    mock_boto.return_value = client

    with pytest.raises(SystemExit):
        register_schema("test-registry", "test-schema", fake_schema, "BACKWARD")


@patch("boto3.client")
def test_fail_on_update_error(mock_boto, fake_schema):
    """Test failure when update also fails."""
    client = MagicMock()
    
    # Mock AlreadyExistsException on create
    create_err = ClientError(
        {"Error": {"Code": "AlreadyExistsException"}}, 
        "CreateSchema"
    )
    client.create_schema.side_effect = create_err
    
    # Mock error on update
    update_err = ClientError(
        {"Error": {"Code": "ValidationException"}}, 
        "UpdateSchema"
    )
    client.update_schema.side_effect = update_err
    mock_boto.return_value = client

    with pytest.raises(SystemExit):
        register_schema("test-registry", "test-schema", fake_schema, "BACKWARD")


@patch("src.register_schema.BOTO3_AVAILABLE", False)
def test_skip_when_boto3_unavailable(fake_schema, capsys):
    """Test graceful handling when boto3 is not available."""
    register_schema("test-registry", "test-schema", fake_schema, "BACKWARD")
    
    captured = capsys.readouterr()
    assert "boto3 not available" in captured.out